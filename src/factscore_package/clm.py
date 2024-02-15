# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16
import json
from huggingface_hub import login
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from .utils import convert_model_to_int8_on_gpu
from .lm import LM

class CLM(LM):
    def __init__(self, model_name, model_dir, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        model_id = 'meta-llama/Llama-2-7b-chat-hf'

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items, you need an access token
        hf_auth = "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"
        model_config = AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
        #model_name_or_path = "TheBloke/Llama-2-70B-chat-AWQ"
        #model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          #trust_remote_code=False, safetensors=True)
        # 2. Tie the weights
        #model.tie_weights()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        self.model = model
        self.tokenizer = tokenizer
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        #self.model = convert_model_to_int8_on_gpu(self.model, device='auto')
        #self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids])
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]

            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]

        return generations, scores

