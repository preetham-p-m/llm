from abc import ABC, abstractmethod


class AiAgent(ABC):
    @abstractmethod
    def generate_content_stream(self, system_prompt: str, user_input: str = ""):
        pass

    @abstractmethod
    def generate_content(self, system_prompt: str, user_input: str = ""):
        pass
