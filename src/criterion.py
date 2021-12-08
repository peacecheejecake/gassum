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


class SimCLSCriterion(_CriterionBase):

    def __init__(
        self,
        model, 
        label_smoothing=0.0,
        ignore_index=-100, 
        reduction='sum', 
    ):
        pass