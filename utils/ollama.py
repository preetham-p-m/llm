import ollama
from ai_agent import AiAgent
import requests
import json


class Ollama(AiAgent):
    def __init__(self):
        self._OLLAMA_API = "http://localhost:11434/api/chat"
        self._HEADERS = {"Content-Type": "application/json"}
        self._MODEL = "deepseek-r1:latest"

    def _generate_response(
        self, system_prompt: str, user_input: str = "", streamed: bool = False
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        payload = {"model": self._MODEL, "messages": messages, "stream": streamed}
        return requests.post(self._OLLAMA_API, json=payload, headers=self._HEADERS)

    def generate_content_stream(self, system_prompt: str, user_input: str = ""):
        return self._generate_response(system_prompt, user_input, streamed=True)

    def generate_stripped_response_string(
        self, system_prompt: str, user_input: str = ""
    ):
        result = self._generate_response(system_prompt, user_input)
        jsonContent = result.json()

        return jsonContent["message"]["content"]
