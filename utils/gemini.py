import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from ai_agent import AiAgent


class Gemini(AiAgent):
    def __init__(self):
        load_dotenv()
        self._client = self._initialize_client()
        self._model = "gemini-2.0-flash"

    def generate_content_stream(self, system_prompt: str, user_input: str = ""):
        messages = [
            Content(role="user", parts=[Part(text=user_input)]),
        ]
        config = self._get_config(system_prompt)

        return self._client.models.generate_content(
            model=self._model, contents=messages, config=config
        )

    def generate_stripped_response_string(
        self, system_prompt: str, user_input: str = ""
    ):
        result = self.generate_content_stream(system_prompt, user_input)

        stream_string = ""
        for candidate in result.candidates:
            for part in candidate.content.parts:
                stream_string += part.text

        return stream_string

    @staticmethod
    def _get_config(system_prompt: str):
        return GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
            system_instruction=[Part.from_text(text=system_prompt)],
        )

    @staticmethod
    def _get_valid_api_key():
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("❌ No API Key found. Please set GEMINI_API_KEY.")

        if api_key.strip() != api_key:
            raise ValueError(
                "⚠️ API Key has spaces at start or end. Please remove them."
            )

        return api_key

    def _initialize_client(self):
        api_key = self._get_valid_api_key()
        return genai.Client(api_key=api_key)
