import argparse
import logging
import os
import datetime

import torch
from torch.utils.data.dataloader import DataLoader

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    BartForConditionalGeneration, 
    BartConfig,
)
from datasets import load_metric

from train import (
    init_wandb, 
    init_optimizer, 
    init_lr_scheduler, 
    prepare_data_loaders,
    load_model,
)
from utils import (
    _postprocess,
    add_arguments_for_config,
    add_arguments_for_generation,
    add_arguments_for_lr_scheduler,
    add_arguments_for_training,
    set_manual_seed_all,
    print_best_model,
    load_data_from_json,
)
from dataset import KobartLabeledDataset
from validate import RougeEvaluator




def train(config):
    if config.wandb:
        wandb_run, wandb_artifact, wandb_text_table = init_wandb(config)

    device = (
        torch.device('cpu') 
        if config.cpu or not torch.cuda.is_available()
        else torch.device('cuda:0')
    )
    
    bart = load_model(config)
    tokenizer_bart = AutoTokenizer.from_pretrained(config.bart_name)
    setattr(tokenizer_bart, 'decoder_start_token_id', bart.config.decoder_start_token_id)
    scorer = AutoModel.from_pretrained(config.scorer_name)
    tokenizer_scorer = AutoTokenizer.from_pretrained(config.scorer_name)

    train_data = load_data_from_json(os.path.join(config.data_dir, 'train_original.json'))
    valid_data = load_data_from_json(os.path.join(config.data_dir, 'valid_original.json'))
    train_loader, valid_loader = prepare_data_loaders(
        config,
        train_data, 
        valid_data,
        tokenizer_bart,
        device,
    )
    optimizer = init_optimizer(config, bart)
    lr_scheduler = init_lr_scheduler(config, optimizer, len(train_loader))








texts = ['최근 들어 딥러닝', '안녕 친구야']
sample_ids = tokenizer(texts, padding=True, return_tensors='pt')['input_ids']

outs = bart.generate(
    sample_ids,
    max_length=50,
    num_beams=16,
    no_repeat_ngram_size=2,
    num_return_sequences=1,
    early_stopping=True,
)

print(outs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--exp_name')
    # parser.add_argument('--tapt')
    parser.add_argument('--bart_name', default='hyunwoongko/kobart')
    parser.add_argument('--sum_batch_size', type=int)
    parser.add_argument('--bart_path', default='hyunwoongko/kobart')
    parser.add_argument('--checkpoint')
    parser.add_argument('--scorer_name', default='kykim/electra-kor-base')
    parser.add_argument('--num_cands', type=int)
    parser.add_argument('--max_input_length', type=int, default=512)
    
    add_arguments_for_training(parser)
    add_arguments_for_generation(parser)
    add_arguments_for_lr_scheduler(parser)
    add_arguments_for_config(parser)

    args = parser.parse_args()
    logging.info(args)

    args.data_dir = f"{args.base_dir}/data"
    args.asset_dir = f"{args.base_dir}/assets"
    args.model_dir = f"{args.base_dir}/models"
    args.checkpoint_dir = f"{args.base_dir}/checkpoints"
    args.exp_name = args.exp_name if args.exp_name is not None else generate_random_name(prefix="kobart")

    set_manual_seed_all(args.seed)

    evaluator, best_model_path = train(args)
    print_best_model(evaluator, best_model_path)