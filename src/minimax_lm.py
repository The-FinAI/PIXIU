from chatlm import ChatLM


# MiniMax models and their context window sizes
MINIMAX_MODELS = {
    "MiniMax-M2.7": 204800,
    "MiniMax-M2.7-highspeed": 204800,
    "MiniMax-M2.5": 204800,
    "MiniMax-M2.5-highspeed": 204800,
}


class MiniMaxLM(ChatLM):
    """Language model class for MiniMax's OpenAI-compatible API.

    MiniMax provides an OpenAI-compatible chat completions endpoint at
    https://api.minimax.io/v1/chat/completions. This class configures
    ChatLM to use MiniMax instead of OpenAI.

    Environment variable: MINIMAX_API_KEY
    """

    API_BASE_URL = "https://api.minimax.io/v1/chat/completions"
    API_KEY_ENV = "MINIMAX_API_KEY"

    @property
    def max_length(self):
        return MINIMAX_MODELS.get(self.model, 204800)

    def _get_temperature(self, temperature):
        """MiniMax requires temperature in (0.0, 1.0]."""
        if temperature <= 0.0:
            return 0.01
        return min(temperature, 1.0)
