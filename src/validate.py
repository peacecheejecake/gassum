import argparse
import logging
import random

import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_metric

from utils import _postprocess, set_manual_seed_all, prepare_train_data
from dataset import KobartLabeledDataset


class RougeEvaluator:

    rouge = load_metric('rouge')

    def __init__(self, config):
        self.config = config
        self._wandb = config.wandb
        self.epoch = 0
        self.start_epoch = 0
        self.best_model_path = None
        self.patience = (
            config.num_epochs 
            if config.patience is None or config.patience <= 0
            else config.patience
        )
        self.patience_count = 0
        
        if self._wandb:
            self.rouge_best_log = {}
        self.init_history()
        self.init_epoch_rouge()
    
    def init_history(self):
        self.history = RougeEvaluator._history_base()

    def init_epoch_rouge(self):
        self.epoch_rouge = RougeEvaluator._epoch_rouge_base()

    def load_history_from_checkpoint(self, checkpoint):
        self.history = checkpoint['history']
        if self._wandb:
            self.rouge_best_log = RougeEvaluator.convert_to_best_rouge(self.history['rouge'])
    
    def load_start_epoch(self, checkpoint):
        if checkpoint.get('epoch') is not None:
            self.start_epoch = checkpoint['epoch']
        elif checkpoint.get('history') is not None:
            self.start_epoch = checkpoint['history']['epoch']
            print(
                "Load epoch count from best model history. "
                "This might not be correct."
            )
        self.epoch = self.start_epoch
    
    @classmethod
    def compute_rouge(cls, summaries, references):
        return cls.rouge.compute(predictions=summaries, references=references)

    def update_epoch_rouge(self, rouge_dict, num_batches):
        for key in self.epoch_rouge:
            self.epoch_rouge[key] += rouge_dict[key].mid.fmeasure / num_batches
    
    def compute_collect_rouge(self, summaries, references, num_batches):
        rouge = RougeEvaluator.compute_rouge(summaries, references)
        self.update_epoch_rouge(rouge, num_batches)
        
    def is_best_model(self, epoch_rouge):
        epoch_rouge_sum = sum(epoch_rouge.values())
        if self.config.save_best_sum:
            is_best_model = epoch_rouge_sum > sum(self.history['rouge'].values())
        else:
            better_counts = len(
                e - h > 0 
                for e, h 
                in zip(epoch_rouge.values(), self.history['rouge'].values())
            )
            is_best_model = better_counts > 1
        return is_best_model

    def check_update_best(self, epoch_rouge):
        if self.is_best_model(epoch_rouge):
            self._update_best(epoch_rouge)
        else:
            self.patience_count += 1

    def _update_best(self, rouge_dict):
        self.history['epoch'] = self.epoch
        self.history['rouge'] = rouge_dict
        self.patience_count = 0
        if self._wandb:
            self.rouge_best_log = self.convert_to_best_rouge(rouge_dict)
    
    def end_of_patience(self):
        return self.patience_count > self.patience

    def end_of_epoch(self):
        _epcoh_rouge = self.epoch_rouge
        self.init_epoch_rouge()
        return _epcoh_rouge

    @classmethod
    def _history_base(cls):
        return {
            'epoch': 0, 
            'rouge': {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}, 
            'patience_count': 0,
            'best_rouge_sums': [],
        }

    @classmethod
    def _epoch_rouge_base(cls):
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    @classmethod
    def convert_to_best_rouge(cls, rouge_dict):
        return {
            **{f'best_{key}': rouge for key, rouge in rouge_dict.items()},
            'best_rouge_sum': sum(rouge_dict.values()),
        }


def validate_epoch(config, model, dataloader, evaluator, epoch=None, wandb_text_table=None, quiet=False, device=None):
    tokenizer = dataloader.dataset.tokenizer
    epoch = epoch if epoch is not None else evaluator.epoch
    random_step = random.randint(1, len(dataloader) - 1)

    if device is not None:
        model.to(device)
    model.eval()

    evaluator.init_epoch_rouge()
    for step, inputs in enumerate(dataloader):
        model_gen = model.generate(
            **inputs['model_inputs'],
            num_beams=config.beam_size,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            early_stopping=True,
            max_length=config.max_gen_length,
        )
        summaries = [
            _postprocess(gen, tokenizer) for gen in tokenizer.batch_decode(model_gen[:, 1:])
        ]
        references = [
            _postprocess(ref, tokenizer) for ref in inputs['labels']
        ]
        evaluator.compute_collect_rouge(summaries, references, len(dataloader))
        
        if wandb_text_table is not None and config.wandb:
            for i, (gen, ref) in enumerate(zip(summaries, references)):
                idx = step * config.valid_batch_size + i
                wandb_text_table.add_data(epoch + 1, idx, gen, ref)

        if not quiet:
            if step in (0, random_step):
                print()
                for gen, ref in zip(summaries[:5], references[:5]):
                    print(f'{gen}\n{ref}\n')
    return evaluator.end_of_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--plm_name', default='hyunwoongko/kobart')
    parser.add_argument('--bos_at_front', action='store_true')
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--max_gen_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    logging.info(args)

    # configs
    device = (
        torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu 
        else torch.device('cpu')
    )

    data_dir = f"{args.base_dir}/data"
    model_dir = f"{args.base_dir}/models"
    
    set_manual_seed_all(args.seed)

    model = BartForConditionalGeneration.from_pretrained(args.plm_name).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/{args.model_name}.pth'))

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    setattr(tokenizer, 'decoder_start_token_id', model.config.decoder_start_token_id)

    _, valid_data = prepare_train_data(
        f"{data_dir}/train_by_agenda_agg.csv", 
        args.valid_ratio, 
        random_split=True,
    )
    valid_dataset = KobartLabeledDataset(valid_data, tokenizer, for_train=False, bos_at_front=args.bos_at_front)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.valid_batch_size, 
        collate_fn=lambda b: valid_dataset.collate(b, device),
    )

    result = validate_epoch(args, model, valid_loader)
    print(result)
