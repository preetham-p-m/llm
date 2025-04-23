import anthropic
from ai_agent import AiAgent
from dotenv import load_dotenv
import os


class Anthropic(AiAgent):
    def __init__(self):
        load_dotenv()
        self._model = "claude-3-7-sonnet-20250219"
        self._client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=self._get_valid_api_key(),
        )

    def generate_content_stream(self, system_prompt: str, user_input: str = ""):
        messages = (
            [
                {"role": "user", "content": user_input},
            ],
        )

        return self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

    def generate_content(self, system_prompt: str, user_input: str = ""):
        messages = (
            [
                {"role": "user", "content": user_input},
            ],
        )

        return self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

    @staticmethod
    def _get_valid_api_key():
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("❌ No API Key found. Please set ANTHROPIC_API_KEY.")

        if api_key.strip() != api_key:
            raise ValueError(
                "⚠️ API Key has spaces at start or end. Please remove them."
            )

        return api_key
