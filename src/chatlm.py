import os
import asyncio
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time

BACKOFF_TIME = 0.1

async def single_chat(client, **kwargs):
    global BACKOFF_TIME
    backoff_time = BACKOFF_TIME
    while True:
        try:
            r = await client.post(**kwargs, timeout=20)
            json_response = r.json()
            s = json_response['choices'][0]["message"]['content']
            time.sleep(backoff_time)
            return s
        except Exception:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time * 30)
            BACKOFF_TIME *= 1.05


async def oa_completion(**kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    import httpx

    async with httpx.AsyncClient() as client:
        tasks = [single_chat(
            client=client,
            url=kwargs["url"], headers=kwargs["headers"],
            json={
                "temperature": kwargs["temperature"], "max_tokens": kwargs["max_tokens"],
                "model": kwargs["model"], "messages": [message,],
            }
        ) for message in kwargs["messages"]]
        results = await asyncio.gather(*tasks)
        return results


class ChatLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, model, truncate=False):
        """

        :param model: str
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        import openai

        self.model = model
        self.truncate = truncate
        # Read from environment variable OPENAI_API_SECRET_KEY
        api_key = os.environ["OPENAI_API_SECRET_KEY"]
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 4096

    @property
    def max_gen_toks(self):
        return 10

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError()

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = "</s>"
            for x in xs:
                if len(ret) >= size:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = "</s>"
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            inps = []
            for context in chunk:
                inps.append(context[0])

            responses = asyncio.run(oa_completion(
                url="https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                model=self.model,
                messages=[{"role": "user", "content": inp} for inp in inps],
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                # stop=until,
            ))

            for resp, context in zip(responses, chunk):
                s = resp

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, "</s>"), s)

                res.append(s)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
