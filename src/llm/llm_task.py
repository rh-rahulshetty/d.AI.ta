from abc import ABC, abstractmethod

from src.modals.llm_data import ModelResponse


class LLMTask(ABC):

    @property
    @abstractmethod
    def tags(self):
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> dict:
        # Takes in data input and returns a key:valued pair items of inputs passed to prompt()
        pass

    @abstractmethod
    def prompt(self, **kwargs) -> str:
        # Takes key:valued argumets from preprocess results and returns the query prompt for LLM
        pass

    @abstractmethod
    def postprocess(self, response: str) -> ModelResponse:
        # Takes in raw string response from LLM and returns a ModelResponse object
        pass
