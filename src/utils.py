import json
import os
import re
from string import ascii_lowercase, digits
import random
import numpy as np
import pandas as pd
import wandb

import torch
from torch.utils.data import Dataset, DataLoader


def set_manual_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(data_path, ignore_nan=False):
    _, ext = os.path.splitext(data_path)
    if ext == '.csv':
        data = pd.read_csv(data_path)
    elif ext == '.json':
        data = pd.read_json(data_path)
    elif ext == '.tsv':
        data = pd.read_csv(data_path, delimiter='\t')
    else:
        raise NotImplementedError('Not supported file type')

    if ignore_nan:
        for key in ('context', 'summary'):
            if key in data.columns:
                data = data.loc[
                    data[key].map(lambda x: not isinstance(x, float))
                ] # remove rows containing missing values
    return data


def load_data_from_json(file_path):
    with open(file_path) as f:
        json_raw = json.load(f)['documents']
    keys = [
        'id', 'title', 'text', 'summary', 'publish_date'
        'size', 'char_count',
        'category', 'media_type', 'media_sub_type', 'media_name',
    ]
    table = {key: [] for key in keys}
    for sample in json_raw:
        for key in keys:
            if key == 'text':
                chunks = []
                for bind in sample['text']:
                    chunks.append(' '.join(s['sentence'] for s in bind))
                table[key].append(' '.join(chunks))
            elif key == 'summary':
                table[key].append(' '.join(sample['abstractive']))
            else:
                table[key].append(sample[key])
    return pd.DataFrame(table, index=table['id'])


def split_train_valid(data, valid_ratio, shuffle=False):
    valid_size = int(valid_ratio * len(data))
    indices = np.arange(len(data))
    if shuffle:
        indices = np.random.permutation(indices)
    train_indices, valid_indices =  indices[:-valid_size], indices[-valid_size:]
    return data.iloc[train_indices], data.iloc[valid_indices]


def prepare_train_data(data_path, valid_ratio, random_split=False, ignore_nan=True):
    data = read_data(data_path, ignore_nan=ignore_nan)
    if valid_ratio > 0:
        # train-valid random split
        return split_train_valid(data, valid_ratio, shuffle=random_split)
    return data


def prepare_kfold_indices(data_path, num_folds, shuffle=False, ignore_nan=True):
    if not isinstance(num_folds, int) or num_folds <= 0:
        raise ValueError('`num_folds` must be a positive integer.')

    data = read_data(data_path, ignore_nan=ignore_nan)
    valid_size = int(len(data) / num_folds)
    indices = np.arange(len(data))
    if shuffle:
        indices = np.random.permutation(indices)
    train_indices, valid_indices = [], []
    for i in range(num_folds):
        sv, ev = i * valid_size, (i + 1) * valid_size
        train_indices.append(np.append(indices[:sv], indices[ev:]).astype('int64'))
        valid_indices.append(indices[sv: ev])
    return data, train_indices, valid_indices


def _postprocess(out, tokenizer):
    out = out.replace(tokenizer.bos_token, '')
    out = out.replace(tokenizer.eos_token, '')
    out = out.replace(tokenizer.pad_token, '')
    out = out.replace('[ ', '[')
    out = out.replace('( ', '(')
    out = out.replace('{ ', '{')
    out = out.replace('< ', '<')
    out = out.replace('\' ', '\'')
    out = out.replace('\" ', '\"')
    out = out.replace(' ]', '>')
    out = out.replace(' )', ')')
    out = out.replace(' }', '}')
    out = out.replace(' >', '>')
    out = out.replace(' \'', '\'')
    out = out.replace(' \"', '\"')
    out = out.replace(' ,', ', ')
    out = out.replace(' .', '. ')
    out = re.sub('[’‘]', '\'', out)
    out = re.sub('[“”]', '\"', out)
    out = out.replace('\\\'', '\'')
    out = out.replace('\\\"', '\"')
    return out.strip()


def preprocess(out):
    out = str(out)
    out = out.replace('[ ', '[')
    out = out.replace('( ', '(')
    out = out.replace('{ ', '{')
    out = out.replace('< ', '<')
    out = out.replace('\' ', '\'')
    out = out.replace('\" ', '\"')
    out = out.replace(' ]', '>')
    out = out.replace(' )', ')')
    out = out.replace(' }', '}')
    out = out.replace(' >', '>')
    out = out.replace(' \'', '\'')
    out = out.replace(' \"', '\"')
    out = out.replace(' ,', ', ')
    out = out.replace(' .', '. ')
    out = re.sub('[’‘]', '\'', out)
    out = re.sub('[“”]', '\"', out)
    return out.strip()


def shift_token_ids_right(token_ids, pad_token_id, start_token_id):
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.LongTensor(token_ids)
    if len(token_ids.shape) == 1:
        token_ids = token_ids.unsqueeze(0)
        
    shifted_token_ids = token_ids.new_zeros(token_ids.shape)
    shifted_token_ids[:, 1:] = token_ids[:, :-1].clone()
    shifted_token_ids[:, 0] = start_token_id
    shifted_token_ids.masked_fill_(shifted_token_ids == -100, pad_token_id)
    return shifted_token_ids


def generate_random_name(length=10, prefix=None, postfix=None):
    chars = ascii_lowercase + digits
    prefix = f'{prefix}_' if prefix is not None else ''
    postfix = f'_{postfix}' if postfix is not None else ''
    if length > 5:
        former = ''.join(random.choice(chars) for _ in range(length // 2))
        latter = ''.join(random.choice(chars) for _ in range(length - length // 2))
        name = f'{former}-{latter}'
    else:
        name = ''.join(random.choice(chars) for _ in range(length))
    return f"{prefix}{name}{postfix}"


def sync_batch_idx(start_epoch, dataloader, sync_wandb=False, org_bsz=None):
    '''
        Compatible only if `epoch * data_size` is divisible of `tgt_bsz`
    '''
    if sync_wandb:
        for epoch in range(start_epoch):
            wandb.log({'epoch': epoch}) # dummy to continue right after last epoch

    tgt_bsz = dataloader.batch_size
    if org_bsz is None:
        org_bsz = tgt_bsz

    class DummyDataset(Dataset):
        def __init__(self):
            self._arr = torch.arange(self.__len__())

        def __len__(self):
            return len(dataloader.dataset)

        def __getitem__(self, index):
            return self._arr[index]

    dummy_loader = DataLoader(
        DummyDataset(),
        batch_size=org_bsz,
        sampler=dataloader.sampler,
    )
    for _ in range(start_epoch):
        for _ in dummy_loader:
            pass
    # last_batch_idx = start_epoch * len(dataloader.dataset)
    # batch_idx = 0
    # while True:
    #     for _ in dataloader:
    #         batch_idx += tgt_bsz
    #         if batch_idx >= last_batch_idx:
    #             return
                

def print_train_info(
        start_epoch, 
        num_epochs, 
        device, 
        model, 
        optimizer, 
        lr_scheduler, 
        criterion=None,
        train_data_size=None, 
        valid_data_size=None, 
        fold=None,
    ):
    print()
    if fold is not None:
        print(f"Start fold #{fold + 1} from epoch {start_epoch + 1}")
    else:
        print(f"Start training from epoch {start_epoch + 1}")
    if train_data_size is not None:
        print(f"- Train Data Size: {train_data_size}")
    if valid_data_size is not None:
        print(f"- Valid Data Size: {valid_data_size}")
    print(f"- Device: {device}")
    print(f"- Model: {model.__class__.__name__}")
    print(f"- Optimizer: {optimizer.__class__.__name__}")
    print(f"- LR Scheduler: {lr_scheduler.__class__.__name__}")
    if criterion is not None:
        print(f"- Criterion: {criterion.__class__.__name__}")
    print(f"- Total Epochs: {num_epochs}")
    print()


def save_checkpoint(epoch, model, optimizer, lr_scheduler, history, path):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'history': history,
    }
    torch.save(checkpoint, path)


def print_best_model(history, final_model_path=None):
    print(f"Best Model")
    print(f"- Epoch: {history['epoch']}")
    rouges = history['rouge']
    print(f"- Rouge-1: {rouges['rouge1']}")
    print(f"- Rouge-2: {rouges['rouge2']}")
    print(f"- Rouge-L: {rouges['rougeL']}")
    if final_model_path is not None:
        print(f"- Path: {final_model_path}")


def add_arguments_for_generation(parser):
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--max_gen_length', type=int, default=512)

def add_arguments_for_lr_scheduler(parser):
    parser.add_argument(
        '--lr_scheduler', 
        default='cosine_decay',
        help=(
            '''
            `cd`: cosine decay, 
            `ca[i]: cosine annealing with cycle i(i is optional), 
            `const`: constant
            '''
        )
    )
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--constant_epochs', type=int, default=0)
    parser.add_argument('--restart_epochs', type=int, default=0)
    parser.add_argument('--warmup_start_lr', type=float, default=0.)
    parser.add_argument('--max_decay_mode', default='exp', help='`exp` or `cos`')


def add_arguments_for_training(parser):
    parser.add_argument(
        '--num_folds', 
        type=int, 
        default=1, 
        help='k for cross validation. If `num_folds` > 1, `valid_ratio` is ignored.',
    )
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--bos_at_front', action='store_true')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--teacher_forcing_decay', type=float, default=0.)
    parser.add_argument('--batch_teacher_forcing', action='store_true')
    parser.add_argument('--no_bias_decay', action='store_true')
    parser.add_argument('--mixout', action='store_true')
    parser.add_argument('--rdrop', action='store_true')
    parser.add_argument('--eda', action='store_true')
    parser.add_argument('--swa_start', type=int)


def add_arguments_for_config(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--save_best_sum', action='store_true')
    parser.add_argument('--num_cands', type=int, default=6)
    parser.add_argument('--off_saving', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')


def load_checkpoint(config, model, optimizer, lr_scheduler, evaluator):
    if config.checkpoint is not None:
        checkpoint_dir = f"{config.base_dir}/checkpoints"
        checkpoint_path = f"{checkpoint_dir}/{config.checkpoint}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if checkpoint.get('model') is not None:
                model.load_state_dict(checkpoint['model'])

                if checkpoint.get('optizmizer') is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                else:
                    print("No optimizer is in the checkpoint.")

                if checkpoint.get('lr_scheduler') is not None:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                else:
                    print("Learning rate scheduler is not in checkpoint.")

                if checkpoint.get('history') is not None:
                    evaluator.load_history_from_checkpoint(checkpoint)
                    print(f"Loaded history: {evaluator.history}.")
                else:
                    print("Best model history is not in checkpoint.")

                if checkpoint.get('epoch') is not None or checkpoint.get('history') is not None:
                    evaluator.load_start_epoch_from_checkpoint(checkpoint)
                else:
                    print("Couldn't found epoch record. Start from epoch 1.")
            else:
                print("Load checkpoint without extra information.")
                model.load_state_dict(checkpoint)
