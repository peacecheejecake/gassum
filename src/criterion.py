import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _smooth_one_hot(labels, smoothing, num_classes):
    return (
        torch.full(
            labels.shape + (num_classes,), 
            smoothing / (num_classes - 1),
        )
        .to(labels.device)
        .scatter_(
            -1, 
            labels.unsqueeze(-1), 
            1 - smoothing,
            reduce='add',
        )
    )


def cross_entropy_with_label_smoothing(logits, labels, smoothing, num_classes, ignore_index=-100):
    with torch.no_grad():
        _labels = labels.masked_fill(labels == ignore_index, 0)
    _labels = (
        _smooth_one_hot(_labels, smoothing, num_classes)
        .masked_fill_(labels.unsqueeze(-1) == ignore_index, 0)
    )
    nll = -F.log_softmax(logits, dim=-1) * _labels
    return nll.sum(-1)


class _CriterionBase(nn.Module):

    def __init__(
        self, 
        model, 
        label_smoothing=0.0,
        ignore_index=-100, 
        reduction='sum', 
    ):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def reduce(self, loss, num_valid):
        if self.reduction == 'mean':
            loss = loss.sum() / num_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def compute_loss(self, *args):
        raise NotImplementedError

    def _forward_model(self, **inputs):
        return self.model(**inputs)

    def get_flattened_logits(self, **inputs):
        return self._forward_model(**inputs).logits.view(-1, self.model.config.vocab_size)


class LabelSmoothingRDropCriterion(_CriterionBase):
    
    def __init__(
        self, 
        model,
        label_smoothing=0.0,
        alpha=0.7,
        ignore_index=-100,
        reduction='sum',
    ):
        super().__init__(
            model=model, 
            label_smoothing=label_smoothing,
            ignore_index=ignore_index, 
            reduction=reduction, 
        )
        self.alpha = alpha

    def compute_kl_loss(self, p1, p2):
        kl_loss_1 = F.kl_div(
            F.log_softmax(p1, dim=-1), 
            F.softmax(p2, dim=-1), 
            reduction='sum',
        )
        kl_loss_2 = F.kl_div(
            F.log_softmax(p2, dim=-1), 
            F.softmax(p1, dim=-1), 
            reduction='sum',
        )
        return (kl_loss_1 + kl_loss_2) / 2

    def compute_ce_loss(self, logits_1, logits_2, labels):
        ce_loss_1 = cross_entropy_with_label_smoothing(
            logits=logits_1, 
            labels=labels, 
            smoothing=self.label_smoothing,
            num_classes=self.model.config.vocab_size,
            ignore_index=self.ignore_index,
        )
        ce_loss_2 = cross_entropy_with_label_smoothing(
            logits=logits_2, 
            labels=labels, 
            smoothing=self.label_smoothing,
            num_classes=self.model.config.vocab_size,
            ignore_index=self.ignore_index,
        )
        
        if self.reduction != 'none':
            ce_loss_1 = ce_loss_1.sum()
            ce_loss_2 = ce_loss_2.sum()
        return (ce_loss_1 + ce_loss_2) / 2

    def compute_loss(self, logits_1, logits_2, labels):
        kl_loss = self.compute_kl_loss(logits_1, logits_2)
        ce_loss = self.compute_ce_loss(logits_1, logits_2, labels)
        loss = self.alpha * kl_loss + ce_loss
        if self.reduction == 'mean':
            loss /= labels.size(0)
        return loss

    def forward(self, model_inputs, labels):
        p1 = self.get_flattened_logits(**model_inputs)
        p2 = self.get_flattened_logits(**model_inputs)
        return self.compute_loss(p1, p2, labels.view(-1))


class LabelSmoothingCrossEntropyCriterion(_CriterionBase):

    def compute_loss(self, logits, labels):
        loss = cross_entropy_with_label_smoothing(
            logits=logits, 
            labels=labels, 
            smoothing=self.label_smoothing,
            num_classes=self.model.config.vocab_size,
            ignore_index=self.ignore_index,
        )
        return self.reduce(loss, (labels != self.ignore_index).sum())

    def forward(self, model_inputs, labels):
        logits = self.get_flattened_logits(**model_inputs)
        return self.compute_loss(logits, labels.view(-1))


class SimCLSRerankCriterion(_CriterionBase):

    def __init__(
        self,
        encoder,
        margin_lambda=0.01,
        label_smoothing=0.0,
        ignore_index=-100, 
        reduction='sum',
    ):
        super().__init__(
            model=encoder, 
            label_smoothing=label_smoothing,
            ignore_index=ignore_index, 
            reduction=reduction, 
        )
        self.encoder = self.model
        self.margin_lambda = margin_lambda

    def rerank(self, matrix, margin=0.):
        upper = matrix.unsqueeze(-1).repeat_interleave(matrix.size(-1), dim=-1)
        lower = matrix.unsqueeze(1).repeat_interleave(matrix.size(-1), dim=1)
        return F.relu((lower - upper + margin).triu(1))

    def forward(self, docs, cands, golds):
        batch_size = docs['input_ids'].size(0)
        num_cands = cands['input_ids'].size(0) // batch_size
        assert num_cands == cands['input_ids'].size(0) // batch_size

        doc_embeddings = self.encoder(**docs)[0][:, 0, :]
        gold_embeddings = self.encoder(**golds)[0][:, 0, :]
        sims_doc_gold = torch.cosine_similarity(doc_embeddings, gold_embeddings, dim=-1)

        cand_embeddings = self.encoder(**cands)[0][:, 0, :].view(batch_size, num_cands, -1)
        doc_embeddings = doc_embeddings.unsqueeze(1).repeat_interleave(num_cands, dim=1)
        sims_doc_cand = torch.cosine_similarity(doc_embeddings, cand_embeddings, dim=-1)

        scores_gold = sims_doc_cand - sims_doc_gold.unsqueeze(1).repeat_interleave(num_cands, dim=1)
        scores_gold = F.relu(scores_gold).sum(1)

        ranks = (
            torch.arange(num_cands)
            .unsqueeze(0)
            .repeat_interleave(batch_size, dim=0)
            .to(scores_gold.device)
        )
        margins = self.rerank(ranks) * self.margin_lambda
        scores_cls = self.rerank(sims_doc_cand, margins).sum((2,1))

        score = (scores_gold + scores_cls).sum()
        if self.reduction == 'mean':
            score /= batch_size
        return score
