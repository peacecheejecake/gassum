import os
import pickle
import argparse
import logging

import pandas as pd

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration, 
)
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from modeling_bart import MixoutBartForConditionalGeneration
from dataset import KobartLabeledDataset, KobartEDADataset
from criterion import LabelSmoothingRDropCriterion, LabelSmoothingCrossEntropyCriterion
from lr_scheduler import (
    ConstantLRScheduler,
    CosineDecayLRScheduler, 
    CosineAnnealingRestartLRScheduler, 
    SimpleLRScheduler,
)
from utils import (
    set_manual_seed_all, 
    add_arguments_for_generation,
    add_arguments_for_lr_scheduler,
    add_arguments_for_training,
    add_arguments_for_config,
    load_data_from_json,
    sync_batch_idx,
    generate_random_name, 
    print_train_info,
    load_checkpoint,
    save_checkpoint,
    print_best_model,
)
from validate import RougeEvaluator, validate_epoch
from info import WANDB_AUTH_KEY, WANDB_PROJECT, WANDB_ENTITY


def init_wandb(config):
    if config.wandb:
        wandb.login(key=WANDB_AUTH_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT, 
            entity=WANDB_ENTITY, 
            config=config, 
            name=config.exp_name,
        )
        wandb_artifact = wandb.Artifact('valid_summary', type='valid_samples')
        wandb_text_table = wandb.Table(columns=['epoch', 'idx', 'generated', 'reference'])
        # wandb.save('./*.py')
        return wandb_run, wandb_artifact, wandb_text_table
    return None, None, None


def init_optimizer(config, model):
    if config.no_bias_decay:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    return optimizer


def init_lr_scheduler(config, optimizer, num_batches):
    if config.lr_scheduler == 'cd':
        lr_scheduler = CosineDecayLRScheduler(
            optimizer=optimizer,
            max_lr=config.lr,
            alpha=config.alpha,
            warmup_steps=num_batches * config.warmup_epochs,
            constant_steps=num_batches * config.constant_epochs,
            total_steps=num_batches * config.num_epochs,
            warmup_start_lr=config.warmup_start_lr,
        )
    elif config.lr_scheduler[:2] == 'ca':
        decay_epochs = (
            config.warmup_epochs 
            if len(config.lr_scheduler) == 2 
            else int(config.lr_scheduler[2:])
        )
        lr_scheduler = CosineAnnealingRestartLRScheduler(
            optimizer=optimizer,
            max_lr=config.lr,
            alpha=config.alpha,
            warmup_steps=num_batches * config.warmup_epochs,
            constant_steps=num_batches * config.constant_epochs,
            restart_steps=num_batches * config.restart_epochs,
            decay_steps=num_batches * decay_epochs,
            total_steps=num_batches * config.num_epochs,
            max_decay_mode=config.max_decay_mode,
            warmup_start_lr=config.warmup_start_lr,
            gamma=config.gamma,
        )
    elif config.lr_scheduler == 'const':
        lr_scheduler = ConstantLRScheduler(
            optimizer=optimizer, 
            lr=config.lr,
            warmup_steps=int(num_batches * config.warmup_epochs),
            warmup_start_lr=config.warmup_start_lr,
        )
    elif config.lr_scheduler == 'simple':
        lr_scheduler = SimpleLRScheduler(
            optimizer=optimizer,
            lr=config.lr,
            warmup_steps=num_batches * config.warmup_epochs,
        )
    return lr_scheduler


def init_criterion(config, model, ignore_id=-100):
    if config.rdrop:
        criterion = LabelSmoothingRDropCriterion(
            model,
            alpha=0.7,
            label_smoothing=config.label_smoothing,
            ignore_index=ignore_id,
            reduction='mean',
        )
    else:
        criterion = LabelSmoothingCrossEntropyCriterion(
            model,
            label_smoothing=config.label_smoothing,
            ignore_index=ignore_id,
            reduction='mean',
        )
    return criterion


def load_model(config, device=None):
    if device is None:
        device = torch.device('cpu')
        
    if config.mixout:
        raise NotImplementedError
        # bart_config = BartConfig.from_pretrained(config.plm_name)
        # model = MixoutBartForConditionalGeneration(bart_config)
    else:
        model = BartForConditionalGeneration.from_pretrained(config.plm_name)
    return model.to(device)


def prepare_data_loaders(config, train_data, valid_data, tokenizer, device=None):
    if config.eda:
        with open(f'{config.asset_dir}/wordnet.pkl', 'rb') as f:
            wordnet = pickle.load(f)
        with open(f'{config.asset_dir}/stopwords.txt') as f:
            stopwords = f.readlines()[0].split()

        train_dataset = KobartEDADataset(
            config,
            train_data, 
            tokenizer=tokenizer, 
            wordnet=wordnet, 
            stopwords=stopwords,
        )
    else:
        train_dataset = KobartLabeledDataset(
            config,
            train_data, 
            tokenizer=tokenizer, 
            for_train=True,
        )
    valid_dataset = KobartLabeledDataset(
        config,
        valid_data, 
        tokenizer, 
        for_train=False, 
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=False,
        collate_fn=lambda b: train_dataset.collate(b, device),
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.valid_batch_size, 
        shuffle=False,
        collate_fn=lambda b: valid_dataset.collate(b, device),
    )
    return train_loader, valid_loader


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--valid_data', required=True)
    parser.add_argument('--exp_name')
    parser.add_argument('--tapt')
    parser.add_argument('--checkpoint')
    parser.add_argument('--plm_name', default='hyunwoongko/kobart')
    
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
    return args


def train_epoch(
    config, 
    criterion, 
    dataloader, 
    optimizer, 
    lr_scheduler, 
    *,
    epoch=None, 
    device=None, 
    quiet=False,
):
    torch.cuda.empty_cache()
    
    if device is not None:
        criterion.model.to(device)
    criterion.model.train()

    epoch_loss = 0
    epoch = epoch if epoch is not None else -1
    for step, inputs in enumerate(dataloader):
        loss = criterion(**inputs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(criterion.model.parameters(), 2.0)
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()
        if not quiet:
            print(
                f'\r{(step + 1) / len(dataloader) * 100:.1f} '
                f'[Epoch {epoch + 1:02d}] (train loss) {loss.item() / config.train_batch_size:.5f}',
                end=''
            )

    dataset = dataloader.dataset
    epoch_loss /= len(dataset)
    if not quiet:
        print(f'\r[Epoch {epoch + 1:02d}] (train loss) {epoch_loss:.5f}', end=' ')

    if config.teacher_forcing_decay > 0 and dataset.tf_ratio > 0:
        dataloader.dataset.tf_ratio = max(0, dataloader.dataset.tf_ratio - config.teacher_forcing_decay)
    return epoch_loss    


def train():
    config = configure()

    if config.wandb:
        wandb_run, wandb_artifact, wandb_text_table = init_wandb(config)

    device = (
        torch.device('cpu') 
        if config.cpu or not torch.cuda.is_available()
        else torch.device('cuda:0')
    )

    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.plm_name)
    setattr(tokenizer, 'decoder_start_token_id', model.config.decoder_start_token_id)

    # train_data = load_data_from_json(os.path.join(config.data_dir, 'train_original.json'))
    # valid_data = load_data_from_json(os.path.join(config.data_dir, 'valid_original.json'))
#     train_data = pd.read_csv(os.path.join(config.data_dir, config.train_data))
#     valid_data = pd.read_csv(os.path.join(config.data_dir, config.valid_data))
    train_data = pd.read_json(os.path.join(config.data_dir, config.train_data))
    valid_data = pd.read_json(os.path.join(config.data_dir, config.valid_data))
    train_loader, valid_loader = prepare_data_loaders(
        config,
        train_data, 
        valid_data,
        tokenizer,
        device,
    )
    optimizer = init_optimizer(config, model)
    lr_scheduler = init_lr_scheduler(config, optimizer, len(train_loader))
    criterion = init_criterion(config, model, ignore_id=train_loader.dataset.ignore_id)

    evaluator = RougeEvaluator(config)
    if config.checkpoint is not None:
        load_checkpoint(config, model, optimizer, lr_scheduler, evaluator)
        sync_batch_idx(evaluator.start_epoch, train_loader, sync_wandb=config.wandb)
    elif config.tapt is not None:
        tapt_state_dict = torch.load(f"{config.asset_dir}/{config.tapt}.pth", map_location='cpu')
        model.load_state_dict(tapt_state_dict)

    model.to(device)
    
    print_train_info(
        evaluator.start_epoch, 
        config.num_epochs, 
        device.type, 
        criterion.model, 
        optimizer, 
        lr_scheduler, 
        criterion,
        len(train_loader.dataset), 
        len(valid_loader.dataset),
    )

    for epoch in range(evaluator.start_epoch, config.num_epochs):
        if evaluator.end_of_patience():
            break
        
        epoch_loss = train_epoch(
            config=config,
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            quiet=False,
        )
        epoch_rouge = validate_epoch(
            config=config,
            model=model,
            dataloader=valid_loader,
            evaluator=evaluator,
            wandb_text_table=wandb_text_table if config.wandb else None,
            epoch=epoch,
            quiet=True,
        )

        evaluator.check_update_best(epoch_rouge)
        if not config.off_saving:
            rouge_str = "+".join([f"{int(r * 1e5)}" for r in epoch_rouge.values()])
            model_out_name = f'{config.exp_name}-{epoch + 1}-{rouge_str}'
            model_out_path = f'{config.model_dir}/{model_out_name}.pth'
            epoch_rouge_sum = sum(epoch_rouge.values())

            if evaluator.is_best_model(epoch_rouge):
                best_model_path = f'{config.model_dir}/{config.exp_name}-best.pth'
                torch.save(model.state_dict(), best_model_path)

            if evaluator.history.get('best_rouge_sums') is None:
                evaluator.history['best_rouge_sums'] = [] # for checkpoints from older version

            best_rouge_sums = evaluator.history['best_rouge_sums']
            need_removal = (
                len(best_rouge_sums) >= config.num_cands and 
                epoch_rouge_sum > best_rouge_sums[-1][0]
            )
            if need_removal:
                _, model_rem_name = best_rouge_sums.pop()
                model_rem_path = f'{config.model_dir}/{model_rem_name}.pth'
                if os.path.exists(model_rem_path):
                    os.remove(model_rem_path)

            if need_removal or len(best_rouge_sums) < config.num_cands:
                best_rouge_sums.append((epoch_rouge_sum, model_out_name))
                torch.save(model.state_dict(), model_out_path)
                best_rouge_sums.sort(reverse=True)

            checkpoint_path = f'{config.checkpoint_dir}/{config.exp_name}-recent.pth'
            save_checkpoint(
                epoch=epoch + 1, 
                model=model, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                history=evaluator.history, 
                path=checkpoint_path,
            )

        if config.wandb:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': epoch_loss, 
                'lr': lr_scheduler.get_last_lr(), 
                **epoch_rouge,
                'rouge_sum': sum(epoch_rouge.values()),
                **evaluator.rouge_best_log,
            })

        for key, value in epoch_rouge.items():
            print(f'({key}) {value:.5f}', end=' ')
        print(f'(best epoch) {evaluator.history["epoch"]}', end='\n\n')

        evaluator.epoch += 1

    # upload wandb artifact (text table)
    if config.wandb and wandb_artifact is not None:
        wandb_artifact.add(wandb_text_table, 'valid_text_all')
        wandb_run.log_artifact(wandb_artifact)

    if best_model_path is not None:
        if not config.off_saving:
            final_model_path = f"{config.model_dir}/{evaluator.history['best_rouge_sums'][0][1]}-best.pth"
            if os.path.exists(best_model_path):
                os.rename(best_model_path, final_model_path)

            model_path_to_remove = f"{config.model_dir}/{evaluator.history['best_rouge_sums'][0][1]}.pth"
            if os.path.exists(model_path_to_remove):
                os.remove(model_path_to_remove)
    else:
        final_model_path = None
    # return evaluator, final_model_path
    print_best_model(evaluator, best_model_path)


if __name__ == '__main__':
    train()
