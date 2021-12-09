import os
import argparse
import logging
import pandas as pd

from torch.optim import lr_scheduler
from transformers import BartTokenizerFast, BartForConditionalGeneration, BartConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from utils import (
    add_arguments_for_config,
    add_arguments_for_lr_scheduler,
    add_arguments_for_training,
    generate_random_name,
    print_train_info, 
    save_checkpoint, 
    prepare_train_data, 
    prepare_kfold_indices, 
    set_manual_seed_all,
    sync_batch_idx,
)
from dataset import KobartDatasetForPreTraining
from train import init_lr_scheduler, init_optimizer
from info import WANDB_AUTH_KEY, WANDB_PROJECT, WANDB_ENTITY


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name')
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--valid_data', required=True)
    parser.add_argument('--plm_name', default='hyunwoongko/kobart')
    parser.add_argument('--checkpoints', nargs='+', default='')
    parser.add_argument('--fill_valid', action='store_true')

    add_arguments_for_config(parser)
    add_arguments_for_training(parser)
    add_arguments_for_lr_scheduler(parser)

    args = parser.parse_args()
    logging.info(args)

    # configs: path, device, name, seed
    data_dir = f"{args.base_dir}/data"
    checkpoint_dir = f"{args.base_dir}/checkpoints"
    asset_dir = f"{args.base_dir}/assets"

    device = (
        torch.device('cpu') 
        if args.cpu or not torch.cuda.is_available()
        else torch.device('cuda:0')
    )
    args.exp_name = (
        args.exp_name if args.exp_name is not None 
        else generate_random_name(length=5)
    )

    set_manual_seed_all(args.seed)

    # wandb configs
    if args.wandb:
        wandb.login(key=WANDB_AUTH_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT, 
            entity=WANDB_ENTITY, 
            config=args, 
            name=args.exp_name,
        )

    # check if all checkpoints are valid
    if args.checkpoints:
        if len(args.checkpoints) != args.num_folds:
            raise ValueError('The number of checkpoints and num of folds are different.')
        for fold, checkpoint in enumerate(args.checkpoints):
            if checkpoint and not os.path.exists(f'{checkpoint_dir}/{checkpoint}.pth'):
                raise ValueError(f'Checkpoint at fold {fold} does not exist.')

    # tokenizer, splitted data/indices
    tokenizer = BartTokenizerFast.from_pretrained(args.plm_name)
    model_config = BartConfig.from_pretrained(args.plm_name)
    setattr(tokenizer, 'decoder_start_token_id', model_config.decoder_start_token_id)

    # train_data_path = f"{data_dir}/{args.data_file}"
    # if args.num_folds > 1:
    #     data_all, train_indices, valid_indices = prepare_kfold_indices(
    #         train_data_path, 
    #         args.num_folds, 
    #         shuffle=False,
    #         ignore_nan=True,
    #     )
    # else:
    #     train_data, valid_data = prepare_train_data(
    #         train_data_path, 
    #         args.valid_ratio, 
    #         random_split=True, 
    #         ignore_nan=True,
    #     )

    train_data = pd.read_csv(os.path.join(data_dir, args.train_data))
    valid_data = pd.read_csv(os.path.join(data_dir, args.valid_data))

    # start of folds
    for fold in range(args.num_folds):
        if args.num_folds > 1: # TODO
            train_data = data_all.iloc[train_indices[fold]]
            valid_data = data_all.iloc[valid_indices[fold]]

        train_dataset = KobartDatasetForPreTraining(
            args,
            train_data,
            tokenizer, 
            fill_short=True,
        )
        valid_dataset = KobartDatasetForPreTraining(
            args,
            valid_data,
            tokenizer, 
            fill_short=args.fill_valid,
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            collate_fn=lambda b: train_dataset.collate(b, device),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.train_batch_size,
            collate_fn=lambda b: valid_dataset.collate(b, device),
        )

        model = BartForConditionalGeneration.from_pretrained(args.plm_name).to(device)
        history = {'epoch': 0, 'train_loss': 1e9, 'valid_loss': 1e9, 'patience_count': 0}

        optimizer = init_optimizer(args, model)
        lr_scheduler = init_lr_scheduler(args, optimizer, len(train_loader))

        if args.checkpoints and args.checkpoints[fold]:
            checkpoint = torch.load(f"{checkpoint_dir}/{args.checkpoints[fold]}.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            history = checkpoint['history']

        start_epoch = history['epoch']
        sync_batch_idx(start_epoch, train_loader)

        print_train_info(
            start_epoch,
            args.num_epochs, 
            device, 
            model, 
            optimizer, 
            lr_scheduler,
            train_data_size=len(train_data), 
            valid_data_size=len(valid_data),
            fold=fold if args.num_folds > 1 else None,
        )

        for epoch in range(start_epoch, args.num_epochs):
            if history['patience_count'] > args.patience:
                break

            model.train()
            train_loss = 0
            for step, inputs in enumerate(train_loader):
                loss = model(**inputs['model_inputs'], labels=inputs['labels']).loss
                train_loss += loss.item()

                lr_scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                print(
                    f'\r{(step + 1) / len(train_loader) * 100:.1f} [Epoch {epoch + 1:02d}] (train loss) {loss.item():.5f}', 
                    end=''
                )

            train_loss /= len(train_loader)
            print(f'\r[Epoch {epoch + 1:02d}] (train loss) {train_loss:.5f}', end=' ')
            
            model.eval()
            valid_loss = 0
            for step, inputs in enumerate(valid_loader):
                with torch.no_grad():
                    loss = model(**inputs['model_inputs'], labels=inputs['labels']).loss
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            print(f'(valid loss) {valid_loss:.5f}')
            
            out_prefix = f'kobart-tapt-{args.exp_name}-{fold}'
            out_postfix = f'{epoch + 1}-{int(valid_loss * 1e5):05d}'
            if history['valid_loss'] > valid_loss:
                history = {
                    'epoch': epoch + 1, 
                    'train_loss': train_loss, 
                    'valid_loss': valid_loss, 
                    'patience_count': 0,
                }
                best_model_path = f'{asset_dir}/{out_prefix}-model-{out_postfix}.pth'
                torch.save(model.state_dict(), best_model_path)
            else:
                history['patience_count'] += 1
            
            checkpoint_path = f'{checkpoint_dir}/{out_prefix}-recent.pth'
            save_checkpoint(
                epoch=epoch + 1, 
                model=model, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                history=history, 
                path=checkpoint_path,
            )

            if args.wandb:
                wandb.log({
                    'train_loss': train_loss, 'valid_loss': valid_loss, 
                    'lr': lr_scheduler.get_lr(),
                    **{f'best_{key}': value for key, value in list(history.items())[:-1]},
                })

        print(history)
