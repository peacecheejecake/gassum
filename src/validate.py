import argparse
import logging
import random
import os
from datetime import datetime
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer

from rouge import Rouge

from utils import (
    _postprocess, 
    set_manual_seed_all, 
    load_data_from_json, 
    add_arguments_for_config, 
    add_arguments_for_generation, 
    add_arguments_for_training,
    print_simple_progress,
)
from dataset import KobartLabeledDataset


class RougeEvaluator:

    rouge = Rouge()

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
    def compute_rouge(cls, predictions, references):
        # return cls.rouge.compute(predictions=summaries, references=references)
        return cls.rouge.get_scores(predictions, references, avg=True)

    def update_epoch_rouge_batch(self, rouge_dict, num_batches):
        for key in self.epoch_rouge:
            # self.epoch_rouge[key] += rouge_dict[key].mid.fmeasure / num_batches
            self.epoch_rouge[key] += rouge_dict[key]['f'] / num_batches
    
    def compute_collect_rouge(self, summaries, references, num_batches):
        score = RougeEvaluator.compute_rouge(summaries, references)
        self.update_epoch_rouge_batch(score, num_batches)
        
    def is_best_model(self, epoch_rouge):
        epoch_rouge_sum = sum(epoch_rouge.values())
        # epoch_rouge_sum = epoch_rouge['rougeL']
        if self.config.save_best_sum:
            is_best_model = epoch_rouge_sum > sum(self.history['rouge'].values())
            # is_best_model = epoch_rouge_sum > self.history['rouge']['rougeL']
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
            'rouge': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}, 
            'patience_count': 0,
            'best_rouge_sums': [],
        }

    @classmethod
    def _epoch_rouge_base(cls):
        return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    
    @classmethod
    def convert_to_best_rouge(cls, rouge_dict):
        return {
            **{f'best_{key}': rouge for key, rouge in rouge_dict.items()},
            'best_rouge_sum': sum(rouge_dict.values()),
            # 'best_rouge_sum': rouge_dict['rougeL'],
        }


def validate_epoch(config, model, dataloader, evaluator=None, epoch=None, wandb_text_table=None, quiet=False, device=None):
    if evaluator is None:
        evaluator = RougeEvaluator(config)

    tokenizer = dataloader.dataset.tokenizer
    epoch = epoch if epoch is not None else evaluator.epoch
    random_step = random.randint(1, len(dataloader) - 1)

    if device is not None:
        model.to(device)
    model.eval()

    # evaluator.init_epoch_rouge()
    predictions, references = [], []
    start_time = datetime.now()
    for step, inputs in enumerate(dataloader):
        if not quiet:
            print_simple_progress(step, total_steps=len(dataloader), start_time=start_time)

        model_gen = model.generate(
            **inputs['model_inputs'],
            num_beams=config.beam_size,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            early_stopping=True,
            max_length=config.max_gen_length,
        )
        _predictions = [
            _postprocess(gen, tokenizer) for gen in tokenizer.batch_decode(model_gen[:, 1:])
        ]
        _references = [
            _postprocess(ref, tokenizer) for ref in inputs['labels']
        ]
        # evaluator.compute_collect_rouge(summaries, references, len(dataloader))
        
        # if wandb_text_table is not None and config.wandb:
        #     for i, (gen, ref) in enumerate(zip(summaries, references)):
        #         idx = step * config.valid_batch_size + i
        #         wandb_text_table.add_data(epoch + 1, idx, gen, ref)

        predictions.extend(_predictions)
        references.extend(_references)

        if not quiet:
            if step in (0, random_step):
                print()
                # for gen, ref in zip(summaries[:5], references[:5]):
                #     print(f'{gen}\n{ref}\n')
                for gen in _predictions[:5]:
                    print(f'{gen}')
                print()

    # return evaluator.end_of_epoch()
    
    rouge_score = {
        key: val['f']
        for key, val
        in Rouge().get_scores(predictions, references, avg=True).items()
    }
    if not quiet:
        print(f'\n{rouge_score}')
        with open('/content/drive/MyDrive/music-bot/tmp_pred_{np.random.randint(100)}.json', 'w') as f:
            json.dump({r: p for r, p in zip(references, predictions)}, f)
    return rouge_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--valid_data', required=True)
    parser.add_argument('--model_name')
    parser.add_argument('--plm_name', default='hyunwoongko/kobart')
    # parser.add_argument('--bos_at_front', action='store_true')
    # parser.add_argument('--valid_ratio', type=float, default=0.2)
    # parser.add_argument('--valid_batch_size', type=int, default=32)
    # parser.add_argument('--beam_size', type=int, default=2)
    # parser.add_argument('--repetition_penalty', type=float, default=1.0)
    # parser.add_argument('--length_penalty', type=float, default=1.0)
    # parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    # parser.add_argument('--max_gen_length', type=int, default=512)
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--cpu', action='store_true')
    add_arguments_for_config(parser)
    add_arguments_for_generation(parser)
    add_arguments_for_training(parser)
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
    if args.model_name is not None:
        model.load_state_dict(torch.load(f'{model_dir}/{args.model_name}.pth'))

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    setattr(tokenizer, 'decoder_start_token_id', model.config.decoder_start_token_id)

    # valid_data = load_data_from_json(os.path.join(data_dir, 'valid_original.json'))
#     valid_data = pd.read_csv(os.path.join(data_dir, args.valid_data))
    valid_data = pd.read_json(os.path.join(data_dir, args.valid_data))
    valid_dataset = KobartLabeledDataset(args, valid_data, tokenizer, for_train=False)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.valid_batch_size, 
        collate_fn=lambda b: valid_dataset.collate(b, device),
    )

    result = validate_epoch(args, model, valid_loader)
    print(result)
