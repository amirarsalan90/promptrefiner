# promptrefiner
### Usage:
* create a file in the root folder named `env_variables.env`. inside the file, include your OpenAI API Key:
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```
* Run the `demo.ipynb` notebook
 
### How to create LLM class that inherits from AbstractLLM

`AbstractLLM` acts as an abstract base class that mandates a consistent interface for LLM implementations. It requires any derived subclass to implement a `predict` method. To adapt `AbstractLLM` for a specific LLM, create a subclass that implements the model's unique interaction mechanisms. Follow these steps:

1. **Define Your Subclass**
   - Inherit from `AbstractLLM` to start your subclass.

2. **Initialize Your Model**
   - Implement the `__init__` method to configure your LLM. This method can include model parameters, authentication details, and other model-specific setups.

3. **Implement the Predict Method**
   - Crucially, define the `predict` method in your subclass. It should take `input_text` and `system_prompt` as arguments and return the LLM's generated output. it can take other parameters as well in addition to `input_text` and `system_prompt`.

### Examples for defining AbstractLLM subclasses:
#### Huggingface model:
```
class LlamaCPPModel(AbstractLLM):
    def __init__(self, checkpoint="HuggingFaceH4/zephyr-7b-beta", temperature=0.1, max_new_tokens=500):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype=torch.float16).to('cuda')
        
    def predict(self, input_text, system_prompt):
        
        messages=[
            {"role": "system", "content": system_prompt},  # Update this as per your needs
            {"role": "user", "content": input_text}
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
        output = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        llm_response = self.tokenizer.decode(output[0][model_inputs.shape[1]:], skip_special_tokens=True)
        return llm_response


llamamodel = LlamaCPPModel(checkpoint="HuggingFaceH4/zephyr-7b-beta", temperature=0.1, max_new_tokens=500)
```

#### llama-cpp-python model:
```
class LlamaCPPModel(AbstractLLM):
    def __init__(self, base_url, api_key, temperature=0.1, max_tokens=200):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def predict(self, input_text, system_prompt):
        response = self.client.chat.completions.create(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},  # Update this as per your needs
            {"role": "user", "content": input_text}
        ],
        temperature=self.temperature,
        max_tokens=self.max_tokens,
    )
        llm_response = response.choices[0].message.content
        return llm_response

llamamodel = LlamaCPPModel(base_url="http://localhost:8000/v1", api_key="sk-xxx", temperature=0.1, max_tokens=400)
```