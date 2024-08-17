# Remoteinference

Simple package to perform remote inference on language models of different providers.

## Getting started
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

