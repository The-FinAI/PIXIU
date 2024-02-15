import numpy as np
import torch
import time
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .lm import LM
from .retrieval import Retrieval

def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

class NPM(LM):

    def __init__(self, bm25, model_name, cache_file):
        assert model_name.startswith("npm")
        self.bm25 = bm25
        self.model_name = model_name
        self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/" + self.model_name)
        self.mask_id = self.tokenizer.mask_token_id

        with open("roberta_stopwords.txt", "r") as f:
            self.stopwords = set()
            for line in f:
                self.stopwords.add(int(line.strip()))

        super().__init__(cache_file=cache_file)

    def load_model(self):
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/" + self.model_name)
        self.model.cuda()
        self.model.eval()

    def save_cache(self):
        super().save_cache()
        self.bm25.save_cache()

    def tokenize(self, texts, skip_special_tokens=False, padding=True):
        assert type(texts)==list
        all_input_ids = self.tokenizer(texts)["input_ids"]
        if skip_special_tokens:
            for i, input_ids in enumerate(all_input_ids):
                assert input_ids[0]==0 and input_ids[-1]==2
                all_input_ids[i] = input_ids[1:-1]
        if not padding:
            return all_input_ids
        max_length = np.max([len(_ids) for _ids in all_input_ids])    
        _all_input_ids = []
        _all_attention_mask = []   
        for i, input_ids in enumerate(all_input_ids):
            n_valid = len(input_ids)
            n_masks = max_length - n_valid
            _all_input_ids.append(input_ids + [0 for _ in range(n_masks)])
            _all_attention_mask.append([1 for _ in range(n_valid)] + [0 for _ in range(n_masks)])
        return torch.LongTensor(_all_input_ids), torch.LongTensor(_all_attention_mask)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)

    def encode(self, texts, skip_special_tokens=False, gt_input_ids=None):
        assert type(texts)==list
        if self.model is None:
            self.load_model()
        if gt_input_ids is not None:
            assert len(texts)==len(gt_input_ids)
        all_input_ids, all_attention_mask = self.tokenize(texts, skip_special_tokens=skip_special_tokens)

        with torch.no_grad():
            outputs = self.model(all_input_ids.cuda(),
                                 all_attention_mask.cuda(),
                                 output_hidden_states=True,
                                 return_dict=True)
            all_logits = outputs["logits"].detach().cpu().numpy()
            all_hidden_states = outputs["hidden_states"][-1].detach().cpu().numpy()

        results = []
        for i, (text, input_ids, logits, hidden_states) in enumerate(zip(texts, all_input_ids, all_logits, all_hidden_states)):
            input_ids = input_ids.numpy().tolist()
            if self.mask_id in input_ids:
                idx = input_ids.index(self.mask_id)
                assert gt_input_ids is not None
                prob = softmax(logits[idx])[gt_input_ids[i]]
                results.append((prob, hidden_states[idx]))
            else:
                _input_ids = [_id for _id in input_ids if _id not in [0, 2]]
                _hidden_states = [h for _id, h in zip(input_ids, hidden_states) if _id not in [0, 2]]
                results.append((_input_ids, _hidden_states))

        return results

    def get_probabilty(self, topic, question):
        passages = self.bm25.get_passages(topic, question, k=3)
        passages = [p["text"].strip() for p in passages]
        cache_key = question + "#" + "#".join(passages)

        if cache_key not in self.cache_dict:
            encoded = self.encode(passages, skip_special_tokens=True)
            stacked_passage_tokens, stacked_passage_vectors = [], []
            for input_ids, vectors in encoded:
                stacked_passage_tokens += input_ids
                if len(vectors)>0:
                    stacked_passage_vectors.append(vectors)
            stacked_passage_vectors = np.concatenate(stacked_passage_vectors, 0)

            question_input_ids = self.tokenize(["Fact: " + question], skip_special_tokens=False, padding=False)[0]
            if 2 in question_input_ids:
                question_input_ids = question_input_ids[:question_input_ids.index(2)]
            question_input_ids = question_input_ids[1:]

            '''
            triples = []
            prefix = True
            for i, input_id in enumerate(question_input_ids):
                if prefix:
                    if input_id==35: # the end of prefix
                        prefix = False
                    continue
                if input_id in [0, 2] or input_id in self.stopwords:
                    continue
                new_question = self.decode(question_input_ids[:i] + [self.mask_id] + question_input_ids[i+1:])
                prob, vector = self.encode(new_question, gt_input_id=input_id)
                triples.append((prob, vector, input_id))
            '''
            triples = []
            batch = []
            gt_input_ids = []
            prefix = True
            for i, input_id in enumerate(question_input_ids):
                if prefix:
                    if input_id==35: # the end of prefix
                        prefix = False
                    continue
                if input_id in [0, 2] or input_id in self.stopwords:
                    continue
                batch.append(self.decode(question_input_ids[:i] + [self.mask_id] + question_input_ids[i+1:]))
                gt_input_ids.append(input_id)
            for (prob, vector), gt_input_id in zip(self.encode(batch, gt_input_ids=gt_input_ids), gt_input_ids):
                triples.append((prob, vector, gt_input_id))

            stacked_question_vectors = np.stack([v for _, v, _ in triples], 0)
            all_scores = np.exp(np.inner(stacked_question_vectors, stacked_passage_vectors) / np.sqrt(stacked_passage_vectors.shape[-1]))

            probs = []
            for (softmax_prob, vector, input_id), scores in zip(triples, all_scores):
                assert len(stacked_passage_tokens)==len(scores)
                if input_id not in stacked_passage_tokens:
                    probs.append(0)
                else:
                    aggregated_scores = defaultdict(list)
                    for token, score in zip(stacked_passage_tokens, scores):
                        aggregated_scores[token].append(score)
                    tot = np.sum([np.sum(v) for v in aggregated_scores.values()])
                    prob = np.sum(aggregated_scores[input_id]) / tot
                    probs.append(prob)

            self.cache_dict[cache_key] = np.mean(probs)
            self.add_n += 1

        return self.cache_dict[cache_key]




