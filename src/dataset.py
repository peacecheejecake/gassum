import random
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import preprocess, shift_token_ids_right


class _DatasetBase(Dataset):

    def __init__(
        self, 
        config,
        data, 
        tokenizer, 
        *,
        for_train,
        labeled,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = config.max_input_length
        self.train = for_train
        self.labeled = labeled
        self.tf_ratio = config.teacher_forcing_ratio
        self.batch_tf = config.batch_teacher_forcing
        self.bos_at_front = config.bos_at_front
        if self.train and not self.labeled:
            raise NotImplementedError
        
        self.config = config
        self.init_special_tokens()

    def init_special_tokens(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.mask_token = tokenizer.mask_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.mask_token_id = tokenizer.mask_token_id
        if hasattr(tokenizer, 'decoder_start_token_id'):
            self.decoder_start_token_id = tokenizer.decoder_start_token_id
        else:
            self.decoder_start_token_id = self.eos_token_id

    def pad(self, token_ids, pad_token_id, max_length):
        """
            `token_ids`: 1d Iterable
        """
        if isinstance(token_ids, torch.Tensor):
            return self.pad_tensor(token_ids, pad_token_id, max_length)
        return list(token_ids) + [pad_token_id] * (max_length - len(token_ids))

    def pad_tensor(self, token_ids, pad_token_id, max_length):
        """
            `tokens`: 1d tensor
        """
        token_length = token_ids.size(0)
        if token_length >= max_length:
            return token_ids
        padded_token_ids = token_ids.new_zeros((max_length,))
        padded_token_ids[:token_length] = token_ids.clone()
        padded_token_ids.masked_fill_(padded_token_ids == 0, pad_token_id)
        return padded_token_ids

    def prepare_decoder_input_ids(self, input_ids, labels):
        return shift_token_ids_right(
            labels if random.random() < self.tf_ratio else input_ids,
            pad_token_id=self.pad_token_id,
            start_token_id=self.decoder_start_token_id,
        )

    def collate(self, batch, device=None):
        if device is None:
            device = self.config.device

        if self.labeled:
            texts, labels = zip(*batch)
        else:
            texts = batch

        def max_token_length(tokens):
            return min(max(map(len, tokens)), self.max_seq_length)

        max_input_length = max_token_length(texts)
        input_ids = []
        for _text_tokens in texts:
            _input_ids = (
                [self.bos_token_id] * int(self.bos_at_front)
                + self.tokenizer.convert_tokens_to_ids(_text_tokens[:max_input_length - int(self.bos_at_front)])
            )
            _input_ids = self.pad(_input_ids, self.pad_token_id, max_input_length)
            input_ids.append(_input_ids)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        model_inputs = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
        }
        
        if self.train:
            max_label_length = max_token_length(labels) if self.tf_ratio == 1 else max_input_length
            label_ids = []
            if not self.batch_tf:
                decoder_input_ids = []
                
            for _input_ids, _label in zip(input_ids, labels):
                if isinstance(_label, str):
                    _label = self.tokenizer.tokenize(_label)
                _label_ids = self.tokenizer.convert_tokens_to_ids(_label[:max_label_length - 1])
                if _label_ids and _label_ids[-1] != self.eos_token_id:
                    _label_ids.append(self.eos_token_id)                    
                _label_ids = self.pad(_label_ids, self.ignore_id, max_label_length)
                label_ids.append(_label_ids)

                if not self.batch_tf:
                    _token_ids = (
                        _label_ids 
                        if random.random() < self.tf_ratio 
                        else self.pad(_input_ids, self.pad_token_id, max_label_length)
                    )
                    _decoder_input_ids = shift_token_ids_right(
                        _token_ids,
                        pad_token_id=self.pad_token_id,
                        start_token_id=self.decoder_start_token_id,
                    )
                    decoder_input_ids.append(_decoder_input_ids)

            labels = torch.LongTensor(label_ids).to(device)
            if not self.batch_tf:
                decoder_input_ids = torch.cat(decoder_input_ids)
            else:
                decoder_input_ids = shift_token_ids_right(
                    labels if random.random() < self.tf_ratio else input_ids,
                    pad_token_id=self.pad_token_id,
                    start_token_id=self.decoder_start_token_id,
                )

            model_inputs['decoder_input_ids'] = decoder_input_ids.to(device)

        return_dict = (
            {'model_inputs': model_inputs, 'labels': labels} 
            if self.labeled
            else model_inputs
        )
        return return_dict

    def __len__(self):
        return len(self.data)


class KobartEDADataset(_DatasetBase):

    def __init__(
        self, 
        config,
        data, 
        tokenizer, 
        *,
        wordnet, 
        stopwords, 
        p_aug=1,
    ):
        super().__init__(
            config=config,
            data=data,
            tokenizer=tokenizer, 
            for_train=True,
            labeled=True,
        )
        self.ignore_id = -100
        self.wordnet = wordnet
        self.stopwords = stopwords
        self.pp = p_aug

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = row['text']
        context = (
            self.eda(context) 
            if random.random() < self.pp
            else preprocess(context)
        )
        summary = preprocess(row['summary'])
        return self.tokenizer.tokenize(context), summary
            
    def eda(self, context):
        temp = context.split(' ')
        alpha = 0.1
        n = int(len(temp) * alpha)
        p = random.random()
        if p < 0.25: # synonym replacement
            for _ in range(n):
                idx = np.random.randint(len(temp))
                if temp[idx] in self.wordnet:
                    synonym = np.random.choice(self.wordnet[temp[idx]])
                    if synonym not in self.stopwords:
                        temp[idx] = synonym
        elif p < 0.5: # random swap
            for _ in range(n):
                idx_1, idx_2 = np.random.randint(len(temp), size=2)
                temp[idx_1], temp[idx_2] = temp[idx_2], temp[idx_1]
        elif p < 0.75: # random insertion
            for _ in range(n):
                word = np.random.choice(temp)
                if word in self.wordnet:
                    synonym = np.random.choice(self.wordnet[word])
                    if synonym not in self.stopwords:
                        idx = np.random.randint(len(temp))
                        temp = temp[:idx] + [synonym] + temp[idx:]
        else:  # random deletion
            _temp = []
            for word in temp:
                if random.random() < alpha:
                    continue
                _temp.append(word)
            temp = _temp
        return preprocess(' '.join(temp))


class KobartLabeledDataset(_DatasetBase):

    def __init__(
        self, 
        config,
        data, 
        tokenizer, 
        *,
        for_train, 
    ):
        super().__init__(
            config=config,
            data=data,
            tokenizer=tokenizer, 
            for_train=for_train,
            labeled=True,
        )
        self.ignore_id = -100
        self.train = for_train

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = preprocess(row['text'])
        summary = preprocess(row['summary'])
        return self.tokenizer.tokenize(context), summary


class KobartEvalDataset(_DatasetBase):

    def __init__(
        self, 
        config,
        data, 
        tokenizer, 
    ):
        super().__init__(
            config=config,
            data=data, 
            tokenizer=tokenizer, 
            for_train=False,
            labeled=False,
        )

    def __getitem__(self, idx):
        context = self.data.iloc[idx]['article']
        context_tokens = self.tokenizer.tokenize(preprocess(context))[:self.max_seq_length]
        return context_tokens


class KobartDatasetForPreTraining(_DatasetBase):

    def __init__(
        self, 
        config,
        data, 
        tokenizer, 
        *,
        mask_prob=0.3,
        fill_short=True,
        permute_scentences=False,
    ):
        super().__init__(
            config=config,
            data=data, 
            tokenizer=tokenizer, 
            for_train=True,
            labeled=True,
        )
        self.ignore_id = -100
        self.mp = mask_prob
        self.fill_short = fill_short
        self.permuting = permute_scentences

    def __getitem__(self, idx):
        label_tokens = self.prepare(idx, self.max_seq_length)
        input_tokens = self.make_noise(label_tokens)
        return input_tokens, label_tokens

    def prepare(self, idx, max_length):
        if max_length <= 0:
            return []
        
        text = self.data.iloc[idx]['text']
        if self.permuting:
            text = self.permute_sentences(text)
        tokens = (
            [self.bos_token] * int(self.bos_at_front)
            + self.tokenizer.tokenize(text)
        )
        
        if self.fill_short:
            jdx = np.random.randint(len(self.data))
            tokens.extend(self.prepare(jdx, max_length - len(tokens)))
        return tokens

    def make_noise(self, tokens):
        i = 0
        input_tokens = []
        while i < len(tokens):
            if (
                random.random() >= self.mp or 
                input_tokens and input_tokens[-1] == self.mask_token
            ):
                input_tokens.append(tokens[i])
                i += 1
            else:
                input_tokens.append(self.mask_token)
                i += np.random.poisson(3)
        return input_tokens

    def permute_sentences(self, text):
        import kss
        sentences = kss.split_sentences(text)
        random.shuffle(sentences)
        return ' '.join(sentences)

    def truncate_random(self, tokens, max_length, start_idx=None):
        if len(tokens) <= max_length:
            return tokens
        if start_idx is None:
            start_idx = np.random.randint(len(tokens) - max_length)
        return tokens[start_idx: start_idx + max_length]


class DatasetForReranker(Dataset):

    def __init__(
        self,
        config,
        data,
        tokenizer,
        *,
        require_golds=False,
    ):
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.require_golds = require_golds

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_dict = {'document': preprocess(row['text']), 'candidates': row['candidates']}
        if self.require_golds:
            input_dict['summary'] = row['summary']
        return input_dict
    
    def __len__(self):
        return len(self.data)

    def collate(self, batch, device=None):
        if device is None:
            device = self.config.device

        docs, cands = [], []
        if self.require_golds:
            golds = []
        for sample in batch:
            docs.append(sample['document'])
            cands.extend(sample['candidates'])
            if self.require_golds:
                golds.append(sample['summary'])
        
        tokenizer_configs = {
            'padding': 'longest',
            'truncation': True,
            'max_length': self.config.max_input_length,
            'return_tensors': 'pt',
        }
        docs = {key: val.to(device) for key, val in self.tokenizer(docs, **tokenizer_configs)}
        cands = {key: val.to(device) for key, val in self.tokenizer(cands, **tokenizer_configs)}
        inputs = (docs, cands)
        if self.require_golds:
            golds = {key: val.to(device) for key, val in self.tokenizer(golds, **tokenizer_configs)}
            inputs += (golds,)
        return inputs
