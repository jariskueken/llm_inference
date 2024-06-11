import logging
import requests
from http import HTTPStatus
from typing import Any

from remoteinference.interfaces.llm import LLMInterface
from remoteinference.util.config import ServerConfig

COMPLETION_ENDPOINT = "completion"
CHAT_ENDPOINT = "v1/chat/completions"

logger = logging.getLogger(__name__)


class LlamaCPPLLM(LLMInterface):
    """
    Implementation if the generic LlmInterface.
    This is for talking to a server hosting a custom LLM
    """

    def __init__(self,
                 cfg: ServerConfig) -> None:
        self.config = cfg

    def completion(self,
                   prompt: str,
                   temperature: float,
                   max_tokens: int,
                   **kwargs) -> Any:
        """
        Send a query to a language model.

        :param prompt: The prompt (string) to send to the model.
        :param temperature: The sampling temperature.
        :param max_tokens: The maximum number of tokens to generate.
        :param kwargs: Additional keyword arguments. See llama.cpp docs for
            possible arguments. Ref:
            https://github.com/ggerganov/llama.cpp/tree/master/examples/server
        Returns:
            str: The model response.
        """

        # construct the json payload for the server request
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            **kwargs
            }

        # "content" is the key to the prompt result in the response json
        response = self.__send_request(payload, COMPLETION_ENDPOINT)
        logger.debug(f"\nPrompt:\n{prompt}\n\nResult:\n{response['content']}")
        return response["content"]

    def chat_completion(self,
                        messages: list[dict[str, str]],
                        temperature: float,
                        max_tokens: int,
                        **kwargs) -> Any:
        """
        Send a query to a chat model.

        :param messages: The messages to send to the model. For the llama.cpp
            webserver this uses the OpenAI format.
        :param stop: A list of stop strings
        :param temperature: The sampling temperature.
        :param max_tokens: The maximum number of tokens to generate.
        :param kwargs: Additional keyword arguments. See llama.cpp docs for
            possible arguments. Ref:
            https://github.com/ggerganov/llama.cpp/tree/master/examples/server
        Returns:
            str: The model response.
        """
        payload = {"messages": messages,
                   "temperature": temperature,
                   "max_tokens": max_tokens,
                   **kwargs}

        logger.debug(f"Sending the following chat completion prompt:\nPrompt:\n{payload}")
        response = self.__send_request(payload, CHAT_ENDPOINT)
        logger.debug(f"\nResult:\n{response}")

        return response

    def __send_request(self,
                       payload: dict[str, Any],
                       api_endpoint: str) -> Any:
        """
        Sends a request to the remote server which is hosting the LLM.

        :param payload: The payload following the llama.cpp format, ref:
            https://github.com/ggerganov/llama.cpp/tree/master/examples/server
        :param api_endpoint: The api endpoint, options are
            (complete, v1/chat/completions)
        Returns:
            Any: The server response in valid json format
        """
        # define request headers
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        server_url = f"http://{self.config.server_address}:\
{self.config.server_port}/{api_endpoint}"
        logger.info(f"Sending request to {server_url}")
        logger.debug(f"\nRaw request content:\n{payload}")
        logger.debug(f"\nRaw request header:\n{headers}")

        # initlialize response object
        response = requests.Response()
        response.status_code = HTTPStatus.NO_CONTENT
        try:
            # send the post request
            response = requests.post(
                url=server_url,
                headers=headers,
                json=payload
                )
            # raise exception if error occurs
            response.raise_for_status()
            logger.info(f"Server response: {response}")
        except requests.RequestException as e:
            logger.error(f"Received an error while trying to access the \
server: {e}")

        if response.content:
            # only return json if we do not have an empty response
            return response.json()
        else:
            return None
