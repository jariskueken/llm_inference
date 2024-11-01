from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class LLMInterface(ABC):
    """
    An abstract interface implementing an Large Language Model inference
    endpoint. All implementations for different types of LLMs should implement
    this interface.
    """
    @abstractmethod
    def completion(self,
                   prompt: str,
                   temperature: float,
                   max_tokens: int,
                   **kwargs) -> str:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            The prompt to generate a completion for.
        temperature : float
            The temperature to use for sampling.
        max_tokens : int
            The maximum number of tokens to generate.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Any
            JSON containing the full return.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_completion(self,
                        messages: list[dict[str, str]],
                        temperature: float,
                        max_tokens: int,
                        **kwargs) -> dict[str, Any]:
        """
        Generate a completion for a chat prompt.

        Parameters
        ----------
        messages : list[dict[str, str]]
            A list of messages in the chat. Each chat element in the list can
            contain a user prompt, as system prompt and an assistan prompt.
        temperature : float
            The temperature to use for sampling.
        max_tokens : int
            The maximum number of tokens to generate.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Any
            JSON containing the full return.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_completion_structured(self,
                                   messages: list[dict[str, str]],
                                   temperature: float,
                                   max_tokens: int,
                                   response_format: BaseModel,
                                   **kwargs) -> Any:
        """
        Generate a completion for a chat prompt while enforcing a custom output
        structure. Note that the API endpoint has to support this. For more
        info ref: https://platform.openai.com/docs/guides/structured-outputs

        Parameters
        ----------
        messages : list[dict[str, str]]
            A list of messages in the chat. Each chat element in the list can
            contain a user prompt, as system prompt and an assistan prompt.
        temperature : float
            The temperature to use for sampling.
        max_tokens : int
            The maximum number of tokens to generate.
        response_format : BaseModel
            A pydantic model that defines the output structure.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Any
            The full server response. The output object of the requested
            structure can be retrieved via response[0].choices.message.parsed
        """
        raise NotImplementedError
