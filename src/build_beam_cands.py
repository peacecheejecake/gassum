import argparse
import logging
import os
from datetime import datetime

import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from rouge import Rouge

from dataset import KobartLabeledDataset
from utils import (
    add_arguments_for_generation,
    add_arguments_for_training,
    add_arguments_for_config,
    set_manual_seed_all,
    _postprocess,
    print_simple_progress,
)


def build_candidates(config, data, device):
    bart = BartForConditionalGeneration(BartConfig.from_pretrained('hyunwoongko/kobart'))
    bart_finetuned = torch.load(config.bart_path, map_location='cpu')
    bart.load_state_dict(bart_finetuned)
    bart = bart.to(device)

    # rouge = load_metric('rouge')
    rouge = Rouge()

    tokenizer = AutoTokenizer.from_pretrained('hyunwoongko/kobart')
    dataset = KobartLabeledDataset(config, data, tokenizer, for_train=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        collate_fn=lambda b: dataset.collate(b, device),
    )

    candidates = []
    bart.eval()
    start_time = datetime.now()
    for step, inputs in enumerate(dataloader):
        print_simple_progress(step, total_steps=len(dataloader), start_time=start_time)
        model_gen = bart.generate(
            **inputs['model_inputs'],
            num_beams=config.beam_size,
            num_return_sequences=config.num_cands,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            early_stopping=True,
            max_length=config.max_gen_length,
        )

        predictions = [
            _postprocess(gen, tokenizer) for gen in tokenizer.batch_decode(model_gen[:, 1:])
        ]
        references = [
            [_postprocess(ref, tokenizer)] * config.num_cands for ref in inputs['labels']
        ]
        for i, _references in enumerate(references):
            _candidates = predictions[i * config.num_cands: (i + 1) * config.num_cands]
            scores = [
                sum(value['f'] for value in score.values())
                for score
                in rouge.get_scores(_candidates, _references) # avg=False
            ]
            candidates.append([c for _, c in sorted(zip(scores, _candidates), reverse=True)])
        print(len(candidates))
        
    data['candidates'] = candidates
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--bart_path', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--bart_name', default='hyunwoongko/kobart')
    parser.add_argument('--max_input_length', type=int, default=512)
    
    add_arguments_for_training(parser)
    add_arguments_for_generation(parser)
    add_arguments_for_config(parser)

    args = parser.parse_args()
    logging.info(args)

    args.data_dir = f"{args.base_dir}/data"
    args.asset_dir = f"{args.base_dir}/assets"
    args.model_dir = f"{args.base_dir}/models"

    set_manual_seed_all(args.seed)
    device = (
        torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu 
        else torch.device('cpu')
    )

    originals= [
        # ('train_w_cands.csv', pd.read_csv('/content/drive/MyDrive/gassum/data/train_original.csv')),
        # ('valid_w_cands.csv', pd.read_csv('/content/drive/MyDrive/gassum/data/valid_original.csv')),
        ('new_test_w_cands.csv', pd.read_csv('/content/drive/MyDrive/gassum/data/new_test.csv')),
    ]
    for out_name, data in originals:
        data = build_candidates(args, data, device)
        data.to_csv(os.path.join(args.data_dir, out_name))
