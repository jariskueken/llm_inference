from abc import ABC, abstractmethod


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
        str
            The completion for the prompt.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_completion(self,
                        messages: list[dict[str, str]],
                        temperature: float,
                        max_tokens: int,
                        **kwargs) -> str:
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
        str
            The completion for the chat prompt.
        """
        raise NotImplementedError