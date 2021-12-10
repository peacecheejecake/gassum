import argparse
import logging
import os
from os import PathLike
from datetime import datetime
from typing import Union

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader

from transformers import (
    AutoTokenizer, 
    AutoModel, 
)

from rouge import Rouge
import wandb

from criterion import SimCLSRerankCriterion
from train import (
    init_optimizer, 
    init_lr_scheduler, 
)
from utils import (
    generate_random_name,
    print_simple_progress,
    print_train_info,
    add_arguments_for_config,
    add_arguments_for_lr_scheduler,
    add_arguments_for_training,
    set_manual_seed_all,
    split_train_valid,
    save_checkpoint,
)
from dataset import DatasetForReranker


class Launcher:

    from info import WANDB_AUTH_KEY, WANDB_PROJECT, WANDB_ENTITY

    def __init__(self, config, device):
        self.prepare_data(config)
        if config.wandb:
            self.init_wandb(config)
        self.load_start(config, device)

    def init_wandb(self, config):
        wandb.login(key=Launcher.WANDB_AUTH_KEY)
        self.wandb_run = wandb.init(
            project=Launcher.WANDB_PROJECT, 
            entity=Launcher.WANDB_ENTITY, 
            config=config, 
            name=config.exp_name,
        )

    def read_data(self, data_dir, data_name):
        path = os.path.join(data_dir, data_name)
        _, ext = data_name.rsplit('.', 1)
        if ext == 'csv':
            data = pd.read_csv(path)
        elif ext =='json':
            data = pd.read_csv(path)
        else:
            raise NotImplementedError
        return data

    def prepare_data(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

        if config.train_data and config.valid_data:
            self.train_data = self.read_data(config.data_dir, config.train_data)
            self.valid_data = self.read_data(config.data_dir, config.valid_data)
        elif config.train_data:
            self.train_data, self.valid_data = split_train_valid(config.train_data, config.valid_ratio, shuffle=True)
        elif config.valid_data:
            self.valid_data = self.read_data(config.data_dir, config.valid_data)
        
        if config.eval_data:
            self.eval_data = self.read_data(config.data_dir, config.eval_data)

        if hasattr(self, 'train_data'):
            train_dataset = DatasetForReranker(config, self.train_data, self.tokenizer, require_golds=True)
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=config.train_batch_size, 
                collate_fn=lambda b: train_dataset.collate(b, device)
            )
        if hasattr(self, 'valid_data'):
            valid_dataset = DatasetForReranker(config, self.valid_data, self.tokenizer, require_golds=False)
            self.valid_loader = DataLoader(
                valid_dataset, 
                batch_size=config.valid_batch_size, 
                collate_fn=lambda b: valid_dataset.collate(b, device)
            )

    def load_start(self, config, device):
        self.encoder = AutoModel.from_pretrained(config.encoder_name).to(device)
        
        if config.mode == 'train':
            self.optimizer = init_optimizer(config, self.encoder)
            self.lr_scheduler = init_lr_scheduler(config, self.optimizer, len(self.train_loader))
            self.criterion = SimCLSRerankCriterion(self.encoder)
            self.history = {
                'epoch': 0, 
                'rouge': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0},
                'best_rouge': {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0},
                'stop_count': 0
            }
        
        if config.checkpoint is not None:
            if os.path.exists(config.checkpoint):
                checkpoint = torch.load(config.checkpoint, map_location=device)
                self.encoder.load_state_dict(checkpoint['model'])
                if config.mode == 'train':
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    self.start_epoch = checkpoint['epoch']
                    self.history = checkpoint['history']
                
                    for epoch in range(self.history['epoch']):
                        if config.wandb:
                            self.wandb_run.log({'epoch': epoch})
                        for _ in self.train_loader:
                            pass

                print(self.history)
            else:
                logging.warn('No checkpoint file exists.')
    

    def train(self, config, device, return_path=False):
        history = self.load_encoder(config.checkpoint)

        print_train_info(
            self.history['epoch'], 
            config.num_epochs, 
            device.type, 
            self.criterion.model, 
            self.optimizer, 
            self.lr_scheduler, 
            self.criterion,
            len(self.train_loader.dataset), 
            len(self.valid_loader.dataset),
        )
        
        best_model_path = None
        for epoch in range(self.history['epoch'], config.num_epochs):
            if history['stop_count'] > config.patience:
                break

            epoch_loss = train_epoch(
                config=config,
                criterion=self.criterion,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=epoch,
            )
            epoch_score = validate_epoch(config, self.encoder, self.valid_loader, quiet=True)

            if sum(epoch_score.values()) > sum(history['best_rouge'].values()):
                history['best_rouge'] = epoch_score
                history['stop_count'] = 0
                is_best_model = True
            else:
                history['stop_count'] += 1
                is_best_model = False
            
            history['epoch'] = epoch + 1
            history['rouge'] = epoch_score

            if not config.off_saving:
                checkpoint_name = f"{config.exp_name}-{epoch}"
                if is_best_model:
                    checkpoint_path = f"{config.checkpoint_dir}/{checkpoint_name}-best.pth"
                    best_model_path = checkpoint_path
                else:
                    checkpoint_path = f"{config.checkpoint_dir}/{checkpoint_name}.pth"
                save_checkpoint(
                    epoch=epoch + 1, 
                    model=self.encoder, 
                    optimizer=self.optimizer, 
                    lr_scheduler=self.lr_scheduler, 
                    history=history, 
                    path=checkpoint_path,
                )

            if config.wandb:
                self.wandb_run.log({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss, 
                    'lr': self.lr_scheduler.get_last_lr(), 
                    **epoch_score,
                    'rouge_sum': sum(epoch_score.values()),
                    **{f"best_{key}": val for key, val in history['best_rouge'].items()},
                    'best_rouge_sum': sum(history['best_rouge'].values()),
                })
            
            for key, value in epoch_score.items():
                print(f"({key}) {value:.5f}", end=' ')
            print(f"(stop count) {history['stop_count']}")
        
        print(
            f"""
                [Best Model]
                - Epoch: {history['epoch'] - history['stop_count']}
                - Rouge-1: {history['best_rouge']['rouge-1']}
                - Rouge-2: {history['best_rouge']['rouge-2']}
                - Rouge-l: {history['best_rouge']['rouge-l']}
                - Saved at: {best_model_path if best_model_path is not None else "Unspecified"}
            """
        )

        if return_path:
            return history, best_model_path
        return history

    def validate(self, config):
        rouge_score = validate_epoch(config, self.encoder, self.valid_loader)
        print(rouge_score)
        return rouge_score
    
    @torch.no_grad()
    def evaluate(self, config, return_path=False):
        self.encoder.eval()

        data = dataloader.dataset.data
        candidates_all = data['candidates']

        batch_size = config.valid_batch_size
        num_cands = len(eval(candidates_all[0])) # assume the same `num_cands` in every sample`
        embedding_size = self.encoder.config.embedding_size

        start_time = datetime.now()
        predictions = []
        for step, (docs, cands) in enumerate(dataloader):
            print_simple_progress(step, total_steps=len(dataloader), start_time=start_time)
            doc_embeddings = (
                self.encoder(**docs)[0][:, 0, :]
                .repeat_interleave(num_cands, dim=0)
                .view(-1, num_cands, embedding_size)
            )
            cand_embeddings = (
                self.encoder(**cands)[0][:, 0, :]
                .view(-1, num_cands, embedding_size)
            )
            scores = torch.cosine_similarity(doc_embeddings, cand_embeddings, dim=-1)
            best_cand_indices = scores.argmax(-1).tolist()
            cand_lists = candidates_all.iloc[batch_size * step: batch_size * (step + 1)]
            predictions.extend(
                eval(cand_list)[idx] 
                for cand_list, idx in zip(cand_lists, best_cand_indices)
            )
        
        submission = pd.read_csv(os.path.join(config.data_dir, 'new_sample_submission.csv'))
        submission['summary'] = predictions
        submission_path = f"{config.submission_dir}/submission {datetime.now()}.csv"
        submission.to_csv(submission_path, index=False)
        print(f"\nSubmission file created: {submission_path}")
        
        if return_path:
            return submission, submission_path
        return submission


def train_epoch(
    config, 
    criterion, 
    dataloader, 
    optimizer, 
    lr_scheduler, 
    *,
    epoch=None, 
):
    criterion.model.train()
    torch.cuda.empty_cache()

    epoch_loss = 0
    epoch = epoch if epoch is not None else -1
    for step, inputs in enumerate(dataloader):
        loss = criterion(*inputs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(criterion.model.parameters(), config.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()
        print(
            f'\r{(step + 1) / len(dataloader) * 100:.1f} '
            f'[Epoch {epoch + 1:02d}] (train loss) {loss.item() / config.train_batch_size:.5f}',
            end=' '
        )
    return epoch_loss


@torch.no_grad()
def validate_epoch(config, encoder, dataloader, *, quiet=False):
    data = dataloader.dataset.data
    candidates_all = data['candidates']

    batch_size = config.valid_batch_size
    num_cands = len(eval(candidates_all[0])) # assume the same `num_cands` in every sample`
    embedding_size = encoder.config.embedding_size

    encoder.eval()
    best_cands = []
    start_time = datetime.now()
    for step, (docs, cands) in enumerate(dataloader):
        if not quiet:
            print_simple_progress(step, total_steps=len(dataloader), start_time=start_time)
        doc_embeddings = (
            encoder(**docs)[0][:, 0, :]
            .repeat_interleave(num_cands, dim=0)
            .view(-1, num_cands, embedding_size)
        )
        cand_embeddings = (
            encoder(**cands)[0][:, 0, :]
            .view(-1, num_cands, embedding_size)
        )
        scores = torch.cosine_similarity(doc_embeddings, cand_embeddings, dim=-1)
        best_cand_indices = scores.argmax(-1).tolist()
        cand_lists = candidates_all.iloc[batch_size * step: batch_size * (step + 1)]
        best_cands.extend(
            eval(cand_list)[idx] 
            for cand_list, idx in zip(cand_lists, best_cand_indices)
        )

    rouge_score = {
        key: val['f']
        for key, val
        in Rouge().get_scores(best_cands, list(data['summary']), avg=True).items()
    }
    if not quiet:
        print(f'\n{rouge_score}')
    return rouge_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--exp_name')
    parser.add_argument('--train_data')
    parser.add_argument('--valid_data')
    parser.add_argument('--eval_data')
    parser.add_argument('--mode', default='train')
    # parser.add_argument('--tapt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--encoder_name', default='kykim/electra-kor-base')

    add_arguments_for_training(parser)
    add_arguments_for_lr_scheduler(parser)
    add_arguments_for_config(parser)

    args = parser.parse_args()
    logging.info(args)

    device = (
        torch.device('cpu') 
        if args.cpu or not torch.cuda.is_available()
        else torch.device('cuda:0')
    )

    args.device = device
    args.data_dir = f"{args.base_dir}/data"
    args.asset_dir = f"{args.base_dir}/assets"
    args.model_dir = f"{args.base_dir}/models"
    args.checkpoint_dir = f"{args.base_dir}/checkpoints"
    args.submission_dir = f"{args.base_dir}/submissions"
    args.exp_name = args.exp_name if args.exp_name is not None else generate_random_name(prefix="kobart-simcls")
    set_manual_seed_all(args.seed)

    launcher = Launcher(args, device)

    if args.mode == 'train':
        assert args.train_data and (args.valid_data or args.valid_ratio)
        launcher.train(args, device)
    elif args.mode == 'eval':
        assert args.eval_data
        launcher.evaluate(args, device)
    elif args.mode == 'valid':
        assert args.valid_data or args.train_data and args.valid_ratio
        launcher.validate(args, device)
    else:
        raise NameError
