# llm_inference

Simple package to perform remote inference on language models of different providers.

## getting started
Install the package
```python
pip install remoteinference
```
If you have a LLM running on a remote server using llama.cpp for example you can initalize the model by running:
```python
from remoteinference.models.models import LlamaCPPLLM
from remoteinference.util.config import ServerConfig

# initalize the model
cfg = ServerConfig(server_address="localhost", server_port=8080)
model = LlamaCPPLLM(cfg)

# run simple completion
completion = model.completion("How is the weather today?",
                               temperature=0.5,
                               max_tokens=50)

```

