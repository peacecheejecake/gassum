from datetime import datetime
import argparse
import json
import os
import pandas as pd

from transformers import (
    AutoTokenizer,
    BartConfig, 
    BartForConditionalGeneration,
)
import torch
from torch.utils.data import DataLoader

from dataset import KobartEvalDataset
from utils import _postprocess, add_arguments_for_config, add_arguments_for_generation, add_arguments_for_training, generate_random_name


def evaluate(args, model, eval_loader):
    start_time = datetime.now()
    summaries = []
    model.eval()
    for step, inputs in enumerate(eval_loader):
        print(f"\r{(step + 1) / len(eval_loader) * 100}% ({str(datetime.now() - start_time)})", end="")
        model_gen = model.generate(
            **inputs,
            num_beams=args.beam_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            early_stopping=args.early_stopping,
            max_length=args.max_gen_length,
        )
        gen_sequences = [
            _postprocess(gen, tokenizer) for gen in tokenizer.batch_decode(model_gen[:, 1:])
        ]
        summaries.extend(gen_sequences)
    print()
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir')
    parser.add_argument('--plm_name')
    parser.add_argument('--model_name')
    parser.add_argument('--max_input_length', type=int, default=512)
    # parser.add_argument('--bos_at_front', action='store_true')
    parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--num_beams', type=int, default=6)
    # parser.add_argument('--repetition_penalty', type=float, default=2.5)
    # parser.add_argument('--length_penalty', type=float, default=1.0)
    # parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
    parser.add_argument('--early_stopping', action='store_true')
    # parser.add_argument('--max_gen_length', type=int, default=512)
    # parser.add_argument('--cpu', action='store_true')
    add_arguments_for_config(parser)
    add_arguments_for_generation(parser)
    add_arguments_for_training(parser)
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    data_dir = os.path.join(args.base_dir, 'data')
    model_dir = os.path.join(args.base_dir, 'models')
    submission_dir = os.path.join(args.base_dir, 'submissions')

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)

    # with open(os.path.join(data_dir, 'test_summary.json')) as f:
    #     test_json = json.load(f)
    # df_keys = [
    #     'original', 'summary', 'passage_id', 
    #     #'doc_name', 'category', 'author', 'publisher', 'publisher_year', 'doc_origin'
    # ]
    # test_dict = {key: [] for key in df_keys}
    # for sample in test_json:
    #     for key in df_keys[:2]:
    #         test_dict[key].append(sample[key])
    #     for key in df_keys[2:]:
    #         test_dict[key].append(sample['Meta'][key])
    # test_data = pd.DataFrame(data=test_dict)

    test_data = pd.read_json(os.path.join(data_dir, 'test_summary.json'))
    dataset = KobartEvalDataset(args, test_data, tokenizer)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=lambda b: dataset.collate(b, device),
    )

    bart_config = BartConfig.from_pretrained(args.plm_name)
    model = BartForConditionalGeneration(bart_config).to(device)

    best_model_path = f"{model_dir}/{args.model_name}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    summaries = evaluate(args, model, data_loader)
    result = pd.DataFrame({
        'Meta': list(test_data['Meta']),
        'summary': summaries,
    })
    prefix = (
        f"{'-'.join(str(datetime.now()).split()).replace(':', '-')}"
        f"_beam{args.beam_size}"
        f"_len{args.max_length}"
        f"_{generate_random_name(3)}"
    )
    output_path = f"{submission_dir}/sub_{prefix}_{args.model_name}.csv"
    result.to_csv(output_path, index=False)

    print(f"Saved at: {output_path}")
    