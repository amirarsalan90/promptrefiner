## AbstractLLM Usage Guide

This section provides instructions on how to utilize the `AbstractLLM` class to integrate various Large Language Models (LLMs) into your Python applications. `AbstractLLM` ensures a standardized interface for different LLMs, allowing easy interchangeability or concurrent use of multiple models.

### Overview of AbstractLLM

`AbstractLLM` acts as an abstract base class that mandates a consistent interface for LLM implementations. It requires any derived subclass to implement a `predict` method, facilitating input text processing and output generation based on given prompts.

### Implementing a Subclass

To adapt `AbstractLLM` for a specific LLM, create a subclass that implements the model's unique interaction mechanisms. Follow these steps:

1. **Define Your Subclass**
   - Inherit from `AbstractLLM` to start your subclass.

2. **Initialize Your Model**
   - Implement the `__init__` method to configure your LLM. This method can include model parameters, authentication details, and other model-specific setups.

3. **Implement the Predict Method**
   - Crucially, define the `predict` method in your subclass. It should take `input_text` and `system_prompt` as arguments and return the LLM's generated output.

### Example: LlamaCPPModel

Below is an example of how to implement a `LlamaCPPModel` subclass for interacting with a specific LLM:

```python
from abstract_llm import AbstractLLM

class LlamaCPPModel(AbstractLLM):
    def __init__(self, base_url, api_key, temperature=0.1, max_tokens=200):
        self.temperature = temperature
        self.max_tokens = max_tokens
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def predict(self, input_text, system_prompt):
        response = self.client.chat.completions.create(
            model="mistral",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        llm_response = response.choices[0].message.content
        return llm_response
```

## Usage
Using your subclass within an application is simple:

```
llama_model = LlamaCPPModel(base_url="http://example.com/api", api_key="your_api_key_here")
output = llama_model.predict("Describe the Eiffel Tower.", "Generate a descriptive text:")
print(output)
```

## Extending AbstractLLM

* Feel free to extend AbstractLLM with additional methods if required by your application, ensuring that any subclass implements at least the essential predict method.
* For more complex interactions with LLMs requiring beyond the input_text and system_prompt, consider extending the parameters of the predict method or configuring these within the __init__ method.