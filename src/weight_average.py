import argparse
from glob import glob

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration
from datasets import load_metric

from utils import (
    add_arguments_for_generation, 
    prepare_train_data, 
    set_manual_seed_all, 
)
from dataset import KobartDatasetForPreTraining, KobartLabeledDataset
from valid import validate_epoch


def average_weights(names, model_dir, device=None, weights=None):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if weights is None:
        weights = [1 / len(names)] * len(names)
    avg_params = {}
    for name, weight in zip(names, weights):
        state_dict = torch.load(f'{model_dir}/{name}.pth', map_location=device)
        for key, param in state_dict.items():
            avg_params[key] = avg_params.get(key, 0) + param * weight
        torch.cuda.empty_cache()
    return avg_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--cands', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--plm_name', default='hyunwoongko/kobart')
    parser.add_argument('--tapt', action='store_true')
    parser.add_argument('--out_prefix')
    add_arguments_for_generation(parser)
    args = parser.parse_args()

    data_dir = f'{args.base_dir}/data'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)

    _, valid_data = prepare_train_data(
        f"{data_dir}/train_by_agenda_agg.csv", args.valid_ratio, random_split=True,
    )

    if args.tapt:
        valid_dataset = KobartDatasetForPreTraining(valid_data, tokenizer)
    else:
        valid_dataset = KobartLabeledDataset(valid_data, tokenizer, for_train=False)
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.valid_batch_size, 
        collate_fn=lambda b: valid_dataset.collate(b, device),
    )

    model = BartForConditionalGeneration(BartConfig.from_pretrained(args.plm_name))
    
    model.to(device)
    model.eval()
    
    if args.tapt:
        model_dir = f'{args.base_dir}/assets'
        best_result = {'loss': 1e9, 'model': None, 'num_weighs': None}
        for num_weights in range(len(args.cands), 0, -1):
            model.load_state_dict(average_weights(
                args.cands[:num_weights], 
                model_dir, 
                device=device,
            ))
            valid_loss = 0
            for step, (input_ids, attention_mask, label_ids) in enumerate(valid_loader):
                model_output = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=label_ids.to(device),
                    return_dict=True,
                )
                loss = model_output['loss']
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            if valid_loss < best_result['loss']:
                best_result = {
                    'loss': valid_loss, 
                    'model': model.state_dict(), 
                    'num_weights': num_weights,
                }
    else:
        model_dir = f'{args.base_dir}/models'
        rouge = load_metric('rouge')
        best_result = {'rouge_sum': 0, 'model': None, 'num_weights': None}
        num_cases = len(args.cands) * (len(args.cands) + 1) // 2
        case_idx = 0
        for num_weights in range(len(args.cands), 0, -1):
            for start_idx in range(len(args.cands) - num_weights + 1):
                case_idx += 1
                print(f"\r{case_idx}/{num_cases}", end='')

                selected = args.cands[start_idx: start_idx + num_weights]
                model.load_state_dict(average_weights(selected, model_dir, device))
                rouge_sum = sum(
                    validate_epoch(args, model, valid_loader, quiet=True)
                    .values()
                )
                if rouge_sum > best_result['rouge_sum']:
                    best_result = {
                        'rouge_sum': rouge_sum, 
                        'model': model.state_dict(), 
                        'num_weights': num_weights,
                        'names': selected,
                    }

    out_name = (
        args.out_prefix 
        if args.out_prefix is not None
        else '-'.join(args.cands[0].split('-')[:-2])
    )
    out_path = f"{model_dir}/{out_name}_avg{best_result['num_weights']}.pth"
    torch.save(best_result['model'], out_path)

    print(f"\n\nBest Result ({best_result['num_weights']} weights):")
    for name in best_result['names']:
        print(f"- {name}")
