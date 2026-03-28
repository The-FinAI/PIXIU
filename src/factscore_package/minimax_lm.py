from .openai_lm import OpenAIModel
import os


class MiniMaxModel(OpenAIModel):
    """MiniMax LLM for FActScore evaluation via OpenAI-compatible API."""

    def __init__(self, model_name="MiniMax-M2.7", cache_file=None):
        key = os.environ.get("MINIMAX_API_KEY", "")
        super().__init__(
            model_name=model_name,
            cache_file=cache_file,
            key=key,
            api_base="https://api.minimax.io/v1",
        )
        # MiniMax requires temperature in (0.0, 1.0]
        self.temp = 0.7

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        message = [{"role": "user", "content": prompt}]
        response = self.call_ChatGPT(
            message, model_name=self.model_name, temp=self.temp, max_len=max_sequence_length
        )
        output = response.choices[0].message.content
        return output, response
