# prompt_refinement

Custom Large Language Model (LLM) Interface Documentation
Overview
This documentation outlines the process for integrating custom Large Language Models (LLMs) into our framework. By adhering to the defined abstract base class, AbstractLLM, users can create adaptable and flexible models that can easily be plugged into our system. This interface standardizes the interaction with various LLMs, ensuring a consistent usage pattern and facilitating the integration of diverse models, such as LlamaCPP, HuggingFace Transformers, or any other LLMs.

AbstractLLM Class
The AbstractLLM class serves as a foundation for creating custom LLM classes. It defines a standard interface that all derived classes must implement, ensuring compatibility with our system.

Methods
__init__(self, *args, **kwargs): Initializes a new instance of an LLM class. This method can accept any number of positional and keyword arguments, making it flexible to accommodate different initialization parameters required by various LLM implementations.

predict(self, input_text, *args, **kwargs): Processes the given input text and returns the model's output. This method must be overridden in all subclasses to implement the specific logic for generating predictions using the underlying LLM. It can accept additional arguments to support various prediction configurations.

Implementing a Custom LLM Class
To integrate a new LLM, create a custom class that inherits from AbstractLLM and implements the required methods. Below is an example implementation using the LlamaCPP model.


Example: LlamaCPPModel

```
class LlamaCPPModel(AbstractLLM):
    def __init__(self, base_url, api_key, *args, **kwargs):
        """
        Initializes the LlamaCPP model with the specified API configuration.
        
        :param base_url: The base URL for the LlamaCPP API.
        :param api_key: The API key for authenticating with the LlamaCPP service.
        """
        super().__init__(*args, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def predict(self, input_text, system_prompt, *args, **kwargs):
        """
        Generates a prediction for the given input text using the LlamaCPP model.
        
        :param input_text: The input text to process.
        :param system_prompt: The system prompt to precede the user's input.
        :return: The LLM's response text.
        """
        response = self.client.chat.completions.create(
            model="mistral",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
        )
        llm_response = response.choices[0].message.content
        return llm_response
```

```
llamamodel = LlamaCPPModel(base_url="http://localhost:8000/v1", api_key="sk-xxx")
response = llamamodel.predict(input_text="what is your name?", system_prompt="you are a very depressed and sarcastic AI")
print(response)
```

Creating Your Custom LLM Class
To create your custom LLM class:

Inherit from AbstractLLM: Your class should inherit from the AbstractLLM class.
Implement __init__ and predict: You must implement the __init__ method to handle any initialization parameters your LLM requires and the predict method to process input text and return the model's output.
Utilize *args and **kwargs: Leverage *args and **kwargs in your methods to maintain flexibility and allow for additional parameters without altering the method signatures.
By following these guidelines, you can ensure that your custom LLM class is compatible with our system and can be easily integrated and utilized.