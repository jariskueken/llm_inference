import logging
import requests
import json
from openai import OpenAI
from together import Together
from http import HTTPStatus
from typing import Any

from remoteinference.interfaces.llm import LLMInterface

COMPLETION_ENDPOINT = "completion"
CHAT_ENDPOINT = "v1/chat/completions"

logger = logging.getLogger(__name__)


class LlamaCPPLLM(LLMInterface):
    """
    Implementation of the generic LLMInterface.
    This is for talking to a server hosting a custom model using llama.cpp.
    To learn more about the llama.cpp server
    ref: https://github.com/ggerganov/llama.cpp
    """

    def __init__(self,
                 server_address: str,
                 server_port: int,
                 api_key: str = "") -> None:
        """
        Initalize the model with the server parameters of the LLamaCPP server.

        :param server_address: The address of the server, contains only the
            host url. If hosted locally just add localhost.
        :param server_port: The port of where to reach the server via TCP as
            we use http protocoll.
        :param api_key: The api key to authenticate with the server. Defaults
            to "" if no key is used.
        """
        self.server_address = server_address
        self.server_port = server_port
        self.api_key = api_key

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
            "Authorization": f"Bearer {self.api_key}"
        }

        server_url = f"http://{self.server_address}:\
{self.server_port}/{api_endpoint}"
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


# TODO: implement retry logic and handle models not suited for completions
# endpoint. Also add validation for the selected model type.
class OpenAILLM(LLMInterface):
    """
    Implementation of the generic LlmInterface.
    This is for talking to the OpenAI API.
    """

    client: OpenAI
    api_key: str
    model: str

    def __init__(self,
                 api_key: str,
                 model: str = 'gpt-4o-mini') -> None:
        """
        Initalize the model with the OpenAI API key.

        :param api_key: The api key to authenticate with the OpenAI API.
        :param model: The model to use for completion. Defaults to
            "gpt-4o-mini". For a list of model options please
            ref: https://platform.openai.com/docs/models,
            ref: https://openai.com/api/pricing/
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

        # TODO: model selection
        self.model = model

    def completion(self,
                   prompt: str,
                   temperature: float,
                   max_tokens: int,
                   **kwargs) -> Any:
        logger.warn("Completion model is currently not properly implemented \
as a there is no guarantee that the selected model is suited for completions. \
Use chat_completion instead.")
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error trying to query the OpenAI API: {e}")

        if response:
            return response["content"]
        else:
            return None

    def chat_completion(self,
                        messages: list[dict[str, str]],
                        temperature: float,
                        max_tokens: int,
                        **kwargs) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error trying to query the OpenAI API: {e}")

        if response:
            return json.loads(response.json())
        else:
            return None


class TogetherAILLM(LLMInterface):
    """
    Implementation of the generic LlmInterface.
    This is for talking to the Together AI API.
    """

    client: Together
    api_key: str

    def __init__(self,
                 api_key: str,
                 model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                 ) -> None:
        """
        Initalize the model with the Together AI API key.

        :param api_key: The api key to authenticate with the Together AI API.
        :param model: The model to use for completion. Defaults to
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo". For a list of model
            options please ref: https://docs.together.ai/docs/chat-models
        """
        self.api_key = api_key
        self.client = Together(api_key=api_key)

        # TODO: model selection
        self.model = model

    # FIXME: completion setup properly
    # ref: https://docs.together.ai/docs/inference-models
    def completion(self,
                   prompt: str,
                   temperature: float,
                   max_tokens: int,
                   **kwargs) -> Any:
        logger.warn("Completion model is currently not properly implemented \
as a there is no guarantee that the selected model is suited for completions. \
Use chat_completion instead.")
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error trying to query the Together AI API: {e}")

        if response:
            return response["content"]
        else:
            return None

    # TODO: does response dump to json work for every model?
    def chat_completion(self,
                        messages: list[dict[str, str]],
                        temperature: float,
                        max_tokens: int,
                        **kwargs) -> dict[str, Any]:

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error trying to query the Together AI API: {e}")

        if response:
            try:
                return json.loads(response.model_dump_json())

            # HACK: Currently hacky solution to handle possible responses which
            # are not json parsable
            except Exception as e:
                logger.warn(f"Error trying to parse the response to json: \
{e}. Return type will differ now.")
                return response
        else:
            return None
