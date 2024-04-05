from typing import List, Union
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, conlist
from typing import List

openai_sys_prompt = """You are an AI assistant who is expert in creating promtps for LLMs. you job is to modify and enhance a prompt for a local smaller LLM model. The systemn prompt to the local LLM model can include some examples that lead the model's behavior.

A number of experiments have been done on different system prompts for the local LLM model. Those experiments which include tested system prompt, tested INPUTs TO Local LLM, and the resulting outputs from the local LLM are provided to you. Your job is to observe the experiments, and come up with a better system prompt for the local LLM to achieve the expected outputs. you can provide some examples, or remove some examples in your suggested system prompt. Remember that total number of examples should be limited to 3, because it adds extra computation and we can't afford it. Pay attention to the fact that, YOU ARE NOT ALLOWED TO USE ANY PARTS OF ANY OF THE EVALUATION INPUTS IN YOUR EXAMPLES, NEVER USE THEM!
"""

class AbstractLLM:
    def __init__(self):
        pass
    
    def predict(self, input_text, system_prompt):
        """
        Process input_text and return the model's output.
        """
        raise NotImplementedError


class EnhancedSystemPrompt(BaseModel):
    Enhanced_System_Prompt: str



class PromptTrackerClass():
    def __init__(self, init_system_prompt: str):
        
        self.target_inputs = []
        self.target_outputs = []

        if not init_system_prompt:
            raise ValueError("init_system_prompt cannot be empty")
        self.llm_system_prompts = [init_system_prompt]

        self.llm_responses = []
    
    def add_evaluation_examples(self, inputs: Union[str, List[str]], outputs: Union[str, List[str]]) -> None:
        # Check if inputs and outputs are both either a list of strings or a single string
        if isinstance(inputs, list) and isinstance(outputs, list):
            if not all(isinstance(i, str) for i in inputs) or not all(isinstance(o, str) for o in outputs):
                raise ValueError("All elements of inputs and outputs lists must be strings.")
            if len(inputs) != len(outputs):
                raise ValueError("Inputs and outputs lists must be of the same length.")
        elif isinstance(inputs, str) and isinstance(outputs, str):
            inputs = [inputs]  # Convert single string input to list
            outputs = [outputs]  # Convert single string output to list
        else:
            raise TypeError("Inputs and outputs must both be either a list of strings or a single string.")
        
        # Add examples
        for input_text, output_text in zip(inputs, outputs):
            self.target_inputs.append(input_text)
            self.target_outputs.append(output_text)
    
    def run_initial_prompt(self, llm_model):
        self.generate_and_add_responses(llm_model)

    def add_system_prompt(self, system_prompt: str) -> None:
        self.llm_system_prompts.append(system_prompt)
    
    def get_last_sys_prompt(self) -> str:
        return self.llm_system_prompts[-1]
    
    
    def add_llm_response(self, response: List[str]) -> None:
        self.llm_responses.append(response)
    
    def generate_and_add_responses(self, llm_model) -> None:
        temp_responses = []
        sys_prompt = self.get_last_sys_prompt()
        for input_text in self.target_inputs:
            output_from_llm = llm_model.predict(input_text=input_text, system_prompt=sys_prompt)
            temp_responses.append(output_from_llm)
        self.add_llm_response(temp_responses)


class OpenaiCommunicator():
    def __init__(self, client, openai_model_code):
        self.client = client
        self.openai_sys_prompt = openai_sys_prompt
        self.openai_model_code = openai_model_code
    
    def create_openai_user_prompt(self, prompttracker):
        total_prompt = ""

        for i in range(len(prompttracker.llm_responses)):
            total_prompt += f"""
Experiment {i}
Experimental System Prompt:
{prompttracker.llm_system_prompts[i]}
"""
            for j in range(len(prompttracker.target_inputs)):
                total_prompt += f"""\n\n

EVALUATION INPUT {j} TO Local LLM:
{prompttracker.target_inputs[j]} \n
EVALUATION OUTPUT {j} FROM Local LLM:
{prompttracker.llm_responses[i][j]} \n
what was EXPECTED to be EVALUATION OUTPUT {j} from Local LLM:
{str(prompttracker.target_outputs[j])} \n\n\n
"""
        return total_prompt
    
    def request_to_openai(self, prompttracker):
        openai_user_prompt = self.create_openai_user_prompt(prompttracker)

        response = self.client.chat.completions.create(
            temperature = 0.1,
            model=self.openai_model_code,
            messages=[
                {"role": "system", "content": self.openai_sys_prompt},
                {"role": "user", "content": openai_user_prompt},
                ],
            functions=[
                {
                "name": "Enhanced_System_Prompt",
                "description": "Enhanced System Prompt for Local LLM",
                "parameters": EnhancedSystemPrompt.model_json_schema()
                }
            ],
            function_call={"name": "Enhanced_System_Prompt"}
        )
        return json.loads(response.choices[0].message.function_call.arguments)['Enhanced_System_Prompt']
    
    def refine_system_prompt(self, prompttracker, llm_model, number_of_iterations):
        for _ in range(number_of_iterations):
            openai_suggestion = self.request_to_openai(prompttracker)
            prompttracker.add_system_prompt(openai_suggestion)
            prompttracker.generate_and_add_responses(llm_model)
