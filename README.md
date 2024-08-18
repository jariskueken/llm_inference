# Remoteinference

Simple package to perform remote inference on language models of different providers.

## Getting Started
Install the package
```python
pip install remoteinference
```

To access an OpenAI model simply import the OpenAILLM and use the chat_completion endpoint to send your contents to the server endpoint. As a response you will receive a valid JSON containing the typicall OpenAI API conform response in a dictionary:
```python
import os

from remoteinference.models import OpenAILLM
from remoteinference.util import user_prompt

model_type = 'gpt-4o-mini'
model = OpenAILLM(
    api_key=os.environ.get('OPEANI_API_KEY'),
    model=model_type
    )

response = model.chat_completion(
    prompt=[user_prompt('Who are you?')],
    temperature=0.5,
    max_tokens=50
)

print(response['choices'][0]['message']['content'])
```

If you have a LLM running on a remote server using [llama.cpp](https://github.com/ggerganov/llama.cpp) you can initalize the model by running:
```python
from remoteinference.models import LlamaCPPLLM
from remoteinference.util import user_prompt

# initalize the model
model = LlamaCPPLLM(
    server_address='localhost',
    server_port=8080
    )

# run simple completion
response = model.chat_completion(
    prompt=[user_prompt('Who are you?')],
    temperature=0.5,
    max_tokens=50
)

print(response['choices'][0]['message']['content'])

```
## Supported Models

### OpenAI
Initialize an OpenAI model by calling:

```python
from remoteinference.models import OpenAILLM

model = OpenAILLM(
    api_key='your_key',
    model='gpt-4o-mini'
)
```

To view a full list of available models for the OpenAI endpoint see [OpenAI docs](https://platform.openai.com/docs/models)


### TogetherAI
Initialize an OpenAI model by calling:

```python
from remoteinference.models import TogetherAILLM

model = TogetherAILLM(
    api_key='your_key',
    model='meta-llama/Llama-3-8b-hf'
)
```

To view a full list of available models for the OpenAI endpoint see [TogetherAI docs](https://docs.together.ai/docs/language-and-code-models)

### LlamaCPP
This package also provides functionality to query a self-hosted language model via [llama.cpp](https://github.com/ggerganov/llama.cpp)

To initalize a model which is hosted locally just do:

```python
from remoteinference.models import LlamaCPPLLM

model = LlamaCPPLLM(
    server_address='localhost',
    server_port=8080
)
```

To see the full specifications of the llama.cpp webserver see [server docs](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).