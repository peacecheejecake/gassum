from datetime import datetime
import argparse
import pandas as pd

from transformers import (
    AutoTokenizer,
    BartConfig, 
    BartForConditionalGeneration,
)
import torch
from torch.utils.data import DataLoader

from dataset import KobartEvalDataset
from utils import _postprocess, generate_random_name


def evaluate(args, model, eval_loader):
    start_time = datetime.now()
    summaries = []
    model.eval()
    for step, inputs in enumerate(eval_loader):
        print(f"\r{(step + 1) / len(eval_loader) * 100}% ({str(datetime.now() - start_time)})", end="")
        model_gen = model.generate(
            **inputs,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            early_stopping=args.early_stopping,
            max_length=args.max_length,
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
    parser.add_argument('--bos_at_front', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_beams', type=int, default=6)
    parser.add_argument('--repetition_penalty', type=float, default=2.5)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--max_gen_length', type=int, default=512)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    data_dir = f'{args.base_dir}/data'
    model_dir = f'{args.base_dir}/models'
    submission_dir = f'{args.base_dir}/submissions'

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)

    test_data = pd.read_csv(f"{data_dir}/test_by_agenda_agg.csv")
    eval_dataset = KobartEvalDataset(test_data, tokenizer, bos_at_front=args.bos_at_front)
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        collate_fn=lambda b: eval_dataset.collate(b, device),
    )

    bart_config = BartConfig.from_pretrained(args.plm_name)
    model = BartForConditionalGeneration(bart_config).to(device)

    best_model_path = f"{model_dir}/{args.model_name}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    summaries = evaluate(args, model, eval_loader)
    result = pd.DataFrame({
        'uid': [f'id_{uid}' for uid in test_data['uid']],
        'summary': summaries,
    })
    prefix = (
        f"{'-'.join(str(datetime.now()).split()).replace(':', '-')}"
        f"_beam{args.num_beams}"
        f"_len{args.max_length}"
        f"_{generate_random_name(3)}"
    )
    output_path = f"{submission_dir}/sub_{prefix}_{args.model_name}.csv"
    result.to_csv(output_path, index=False)

    print(f"Saved at: {output_path}")
    