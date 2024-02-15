import pickle
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16
import json
from huggingface_hub import login


class LM():

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.save_interval = 100
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()
        #model_id = 'meta-llama/Llama-2-7b-chat-hf'

        #bnb_config = BitsAndBytesConfig(
            #load_in_4bit=True,
            #bnb_4bit_quant_type='nf4',
            #bnb_4bit_use_double_quant=True,
            #bnb_4bit_compute_dtype=bfloat16
        #)

        # begin initializing HF items, you need an access token
        #hf_auth = "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"
        #model_config = AutoConfig.from_pretrained(
           # model_id,
            #use_auth_token=hf_auth
        #)

        #model = AutoModelForCausalLM.from_pretrained(
            #model_id,
            ##trust_remote_code=True,
            #config=model_config,
            #quantization_config=bnb_config,
            #device_map='auto',
            #use_auth_token=hf_auth
        #)
        #model_name_or_path = "TheBloke/Llama-2-70B-chat-AWQ"
        #model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          #trust_remote_code=False, safetensors=True)
        # 2. Tie the weights
        #model.tie_weights()

        #tokenizer = AutoTokenizer.from_pretrained(
            #model_id,
            #use_auth_token=hf_auth
        #)
        # 3. Create the pipeline using the model with tied weights.
        #generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        #self.model = generator


    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()

        if prompt.endswith(" True or False?\nAnswer:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated

    """
    def _generate(self, prompt, max_output_length):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        generate_kwargs = dict(max_new_tokens=max_output_length, do_sample=True, temperature=0.5)
        output = self.model(prompt, **generate_kwargs)
        #print(output)
        output = output[0]['generated_text'][len(prompt):].strip()
        if "\n" in output:
            output = output[:output.index("\n")]
        return output
    """
    def save_cache(self):
        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache



