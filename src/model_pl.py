import torch
import pytorch_lightning as pl
from utils import _postprocess
from datasets import load_metric

class BartForConditionalGenerationOnLightning(pl.LightningModule):
    
    def __init__(
        self, model, optimizer, lr_scheduler, tokenizer, train_loader, valid_loader, verbose_end=3,
    ):
        super().__init__()
        self.model = model
        print(model.__class__)
        self.rouge = load_metric('rouge')
        self.verbose_max_idx = int(verbose_end)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        self.patience_count = 0
        self.history = {'epoch': 0, 'rouge': self.epoch_rouge, 'patience_count': 0}
        self.epoch_count = 0
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def load_checkpoint_manual(self, checkpoint):
        if checkpoint.get('model') is not None:
            self.model.load_state_dict(checkpoint['model'])
            if checkpoint.get('lr_scheduler') is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                print("No lr scheduler checkpoint found")
            if checkpoint.get('best_model') is not None:
                self.history = self.history['best_model']
                self.patience_count = self.history['patience_count']
                self.epoch_rouge = self.history['rouge']
                self.epoch_count = self.history['epoch']
            else:
                print("No best model history found")
        else:
            print("No extra information on checkpoint")
            self.model.load_state_dict(checkpoint)

    def forward(self, input_ids, attention_mask, labels):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return model_output['loss']

    def generate(
        self, 
        input_ids, 
        attention_mask, 
        num_beams, 
        repetition_penalty,
        length_penalty,
        no_repeat_ngram_size,
        early_stopping,
        max_length,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            max_length=max_length,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}

    def training_step(self, batch, batch_idx):
        return self(*batch)

    def validation_step(self, batch, batch_idx):
        text_input_ids, text_attention_mask, references = batch
        model_gen = self.model.generate(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            no_repeat_ngram_size=2,
            early_stopping=True,
            max_length=512,
        )
        gen_sequences = [
            _postprocess(gen, self.tokenizer) for gen in self.tokenizer.batch_decode(model_gen[:, 1:])
        ]
        references = [
            _postprocess(ref, self.tokenizer) for ref in references
        ]

        rouge = self.rouge.compute(predictions=gen_sequences, references=references)
        for key in self.epoch_rouge:
            self.epoch_rouge[key] += rouge[key].mid.fmeasure
        
        if batch_idx <= self.verbose_max_idx:
            for gen, ref in zip(gen_sequences[:5], references[:5]):
                print(f'{gen}\n{ref}\n')

        self.valid_size = batch_idx

    def validation_epoch_end(self, outputs):
        self.epoch_count += 1

        self.epoch_rouge = {key: val / self.valid_size for key, val in self.epoch_rouge.items()}
        if sum(self.epoch_rouge.values()) > sum(self.history['rouge'].values()):
            self.history = {'epoch': self.epoch_count, 'rouge': self.epoch_rouge, 'patience_count': self.patience_count}
        else:
            self.patience_count += 1

        checkpoint = {
            'model': self.model.state_dict(), 
            'lr_scheduler': self.lr_scheduler.state_dict(), 
            'best_model': self.history,
        }
        rouge_str = "+".join([f"{int(r * 1e5)}" for r in self.epoch_rouge.values()])
        torch.save(checkpoint, f"./kobart-hunmin-{self.epoch_count}-{rouge_str}.pth")

        print(self.history)
        self.epoch_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    def test_step(self, batch, batch_idx):
        pass
