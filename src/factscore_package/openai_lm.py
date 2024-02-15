from .lm import LM
from openai import OpenAI
import openai
import sys
import time
import os
import numpy as np
import logging

#os.environ["http_proxy"] = "http://localhost:27890"
#os.environ["https_proxy"] = "http://localhost:27890"

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key=""):
        self.model_name = model_name
        self.temp = 0.7
        self.save_interval = 100
        self.client = OpenAI(api_key=key.strip())
        super().__init__(cache_file)

    def load_model(self):
        pass
        # load api key
        #key_path = self.key_path
        #assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        #with open(key_path, 'r') as f:
            #api_key = f.readline()
        #self.client = OpenAI(api_key=api_key.strip())
        #self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = self.call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            output = response.choices[0].message.content
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = self.call_GPT3(prompt, temp=self.temp)
            # Get the output from the response
            output = response.choices[0].text
            return output, response
        else:
            raise NotImplementedError()

    def call_ChatGPT(self, message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = self.client.chat.completions.create(model=model_name,
                                                        messages=message,
                                                        max_tokens=max_len,
                                                        temperature=temp)
                received = True
            except:
                # print(message)
                num_rate_errors += 1
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    assert False

                logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
                time.sleep(np.power(2, num_rate_errors))
        return response


    def call_GPT3(self, prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = self.client.completions.create(model="gpt-3.5-turbo-instruct",
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo)
                received = True
            except:
                error = sys.exc_info()[0]
                num_rate_errors += 1
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))
                time.sleep(np.power(2, num_rate_errors))
        return response
