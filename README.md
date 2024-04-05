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
