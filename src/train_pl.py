import os
import pickle
import argparse
import pandas as pd
import numpy as np
from transformers import BartConfig, BartForConditionalGeneration
from kobart import get_kobart_tokenizer
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from modeling_bart import MixoutBartForConditionalGeneration
from model import BartForConditionalGenerationOnLightning
from dataset import KobartDataset, KobartEDADataset, KobartEvalDataset
from lr_scheduler import CosineDecayWithWarmup


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='.')
parser.add_argument('--tapt_name', type=str, default='')
parser.add_argument('--checkpoint_name', type=str, default='')
parser.add_argument('--bart_name', type=str, default='hyunwoongko/kobart')
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--mixout', type=bool, default=False)
parser.add_argument('--eda', type=bool, default=False)
parser.add_argument('--exp_name', type=str, default='lg_dialog_summarization')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=bool, default=True)
args = parser.parse_args()

print(args)

# --------- config ---------
device = torch.device('cuda:0') if torch.cuda.is_available() and args.gpu else torch.device('cpu')

data_dir = f"{args.base_dir}/data"
asset_dir = f"{args.base_dir}/assets"
checkpoint_dir = f"{args.base_dir}/checkpoints"
submission_dir = f"{args.base_dir}/submissions"

# os.environ['HOROVOD_WITH_PYTORCH'] = '1'
# ---------------------------

# --------- seed ---------
pl.seed_everything(args.seed)
# ---------------------------

train_data = pd.read_csv(f"{data_dir}/train_by_agenda_agg.csv")
test_data = pd.read_json(f"{data_dir}/test.json")

# remove missing values
train_data = train_data.loc[train_data.summary.map(lambda x: not isinstance(x, float))]

# train-valid random split
valid_size = int(args.valid_ratio * len(train_data))
train_indices = np.random.permutation(np.arange(len(train_data)))
train_indices, valid_indices = train_indices[:-valid_size], train_indices[-valid_size:]
train_data, valid_data = train_data.iloc[train_indices], train_data.iloc[valid_indices]

# load kobart tokenizer
tokenizer = get_kobart_tokenizer()

# prepare dataset
if args.eda:
    with open(f'{asset_dir}/wordnet.pkl', 'rb') as f:
        wordnet = pickle.load(f)
    with open(f'{asset_dir}/stopwords.txt') as f:
        stopwords = f.readlines()[0].split()
    train_dataset = KobartEDADataset(train_data, tokenizer, wordnet, stopwords)
else:
    train_dataset = KobartDataset(train_data, tokenizer, for_train=True)
valid_dataset = KobartDataset(valid_data, tokenizer, for_train=False)
eval_dataset = KobartEvalDataset(test_data, tokenizer)

train_loader = DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_dataset.collate,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=valid_dataset.collate,
)
eval_loader = DataLoader(
    eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.collate,
)

if args.mixout:
    bart_config = BartConfig.from_pretrained(args.bart_name)
    bart = MixoutBartForConditionalGeneration(bart_config)
else:
    bart = BartForConditionalGeneration.from_pretrained(args.bart_name)

optimizer = torch.optim.AdamW(bart.parameters(), lr=args.lr)
lr_scheduler = CosineDecayWithWarmup(
    optimizer=optimizer,
    warmup_steps=len(train_loader) * args.warmup_epochs,
    total_steps=len(train_loader) * args.num_epochs,
    max_lr=args.lr,
)
model = BartForConditionalGenerationOnLightning(
    bart, 
    optimizer, 
    lr_scheduler, 
    tokenizer,
    train_loader,
    valid_loader,
)

if args.tapt_name:
    tapt_state_dict = torch.load(f"{data_dir}/{args.tapt_name}.pth")
    model.model.load_state_dict(tapt_state_dict)

    # for name, param in model.model.named_parameters():
    #     if name[:5] == 'model':
    #         param.data = tapt_state_dict[name]

# model = MixoutBartForConditionalGeneration.from_pretrained('hyunwoongko/kobart').to(device)
# model = MixoutBartForConditionalGeneration(BartConfig.from_pretrained('hyunwoongko/kobart')).to(device)

# no_decay = ["bias", "LayerNorm.weight"]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": 0.01,
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]
# optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

if args.checkpoint_name:
    checkpoint_path = f"{data_dir}/{args.checkpoint_name}.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_checkpoint_manual(checkpoint)
    
trainer = pl.Trainer(
    # enable_progress_bar=True,
    gpus=torch.cuda.device_count(),
    # accelerator='horovod',
)

start_epoch = model.history['epoch']
for epoch in range(start_epoch, args.num_epochs):
    trainer.fit(model, train_loader)
    trainer.validate(model, valid_loader, verbose=True)

# start_epoch = history['epoch']
# for epoch in range(start_epoch, num_epochs):
#     if history['patience_count'] > patience:
#         break

#     model.train()
#     epoch_loss = 0
#     for step, inputs in enumerate(train_loader):
#         articles, references = inputs
#         input_ids, attention_mask = articles.input_ids.to(device), articles.attention_mask.to(device)
#         labels = references.input_ids.to(device)
#         labels[labels == tokenizer.pad_token_id] = train_dataset.ignore_id
        
#         model_output = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             return_dict=True,
#         )
        
#         loss = model_output['loss']
#         epoch_loss += loss.item()

#         lr_scheduler.step()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f'\r{(step + 1) / len(train_loader) * 100:.1f} [Epoch {epoch + 1:02d}] (train loss) {loss.item():.5f}', end='')

#     epoch_loss /= len(train_loader)
#     print(f'\r[Epoch {epoch + 1:02d}] (train loss) {epoch_loss:.5f}')
    
#     model.eval()
#     epoch_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
#     for step, inputs in enumerate(valid_loader):
#         if step > len(valid_loader) / 2:
#             break
#         articles, references = inputs
#         input_ids, attention_mask = articles.input_ids.to(device), articles.attention_mask.to(device)
#         with torch.no_grad():
#             model_gen = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 num_beams=2,
#                 repetition_penalty=2.5,
#                 length_penalty=1.0,
#                 no_repeat_ngram_size=2,
#                 early_stopping=True,
#                 max_length=512,
#             )
#         gen_sequences = [
#             _postprocess(gen, tokenizer) for gen in tokenizer.batch_decode(model_gen[:, 1:])
#         ]
#         references = [
#             _postprocess(ref, tokenizer) for ref in references
#         ]
#         _rouge = rouge.compute(predictions=gen_sequences, references=references)
#         for key in epoch_rouge:
#             epoch_rouge[key] += _rouge[key].mid.fmeasure

#         if step == 0:
#             for gen, ref in zip(gen_sequences[:5], references[:5]):
#                 print(f'{gen}\n{ref}\n')

#     epoch_rouge = {key: value / len(valid_loader) for key, value in epoch_rouge.items()}
#     if any([e > h for e, h in zip(epoch_rouge.values(), history['rouge'].values())]):
#         history['epoch'] = epoch + 1
#         history['rouge'] = epoch_rouge
#         history['patience_count'] = 0
#         saving_dir = '/content/drive/MyDrive/open'
#     else:
#         history['patience_count'] += 1
#         saving_dir = '/content'
    
    # rouge_str = "+".join([f"{int(r * 1e5)}" for r in epoch_rouge.values()])
    # saving_path = f'{saving_dir}/kobart-{exp_idx}-{epoch + 1}-{rouge_str}.pth'
    # torch.save(model.state_dict(), saving_path)

    # print(history)
    # print()
