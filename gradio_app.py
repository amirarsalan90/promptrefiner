from typing import Dict, List, Type, Union
from typing import List
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

import sglang as sgl
from pydantic import BaseModel, conlist
from typing import List
from sglang.srt.constrained import build_regex_from_object
import gradio as gr
import ast

load_dotenv("env_variable.env")
client = OpenAI()


class ConceptsList(BaseModel):
    #the list name has an important effect on the response! choose it wisely!
    Concepts_List: conlist(str, max_length=10)


class EnhancedSystemPrompt(BaseModel):
    #the list name has an important effect on the response! choose it wisely!
    Enhanced_System_Prompt: str


@sgl.function
def pydantic_gen_ex(s, list_element):
    s += list_element
    s += sgl.gen(
        "",
        max_tokens=1024,
        temperature=0,
        regex=build_regex_from_object(ConceptsList),  # Requires pydantic >= 2.0
    )

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))


def create_mistral_total_prompt(system_prompt, input):
    final_input = f"""text:{input}
Concepts_List:"""    
    final_prompt = system_prompt + "\n" + final_input

    return final_prompt


def local_llm_call(input):
    state = pydantic_gen_ex.run(input)
    return str(json.loads(state.text()[len(input):])["Concepts_List"])



@dataclass
class PromptsClass:
    """Class for keeping track of an item in inventory."""
    mistral_system_prompts: List[str]
    target_input: List[str]
    target_output: List[str]
    mistral_responses: List[List[str]]


promptTracker = PromptsClass([],[],[],[])



def create_openai_user_prompt(prompttracker):
    total_prompt = ""

    for i in range(len(prompttracker.mistral_responses)):
        total_prompt += f"""\n\n\n\n
Experiment {i}
Mistral System Prompt:
{prompttracker.mistral_system_prompts[i]}
"""
        for j in range(len(prompttracker.target_input)):
            total_prompt += f"""\n\n

EVALUATION INPUT {j} TO MISTRAL:
{prompttracker.target_input[j]}
EVALUATION OUTPUT {j} FROM MISTRAL:
{prompttracker.mistral_responses[i][j]}
what was EXPECTED to be EVALUATION OUTPUT {j} from MISTRAL:
{str(prompttracker.target_output[j])} \n\n
"""
        
    return total_prompt




def request_to_openai(prompttracker, openai_sys_prompt):
    openai_user_prompt = create_openai_user_prompt(prompttracker=prompttracker)
    
    response = client.chat.completions.create(
        temperature = 0.1,
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": openai_sys_prompt},
            {"role": "user", "content": openai_user_prompt},
            ],
        functions=[
            {
            "name": "Enhanced_System_Prompt",
            "description": "Enhanced System Prompt for Mistral LLM",
            "parameters": EnhancedSystemPrompt.model_json_schema()
            }
        ],
        function_call={"name": "Enhanced_System_Prompt"}
    )
    return json.loads(response.choices[0].message.function_call.arguments)['Enhanced_System_Prompt']


def refine_system_prompt_with_gpt4(number_of_iterations, prompttracker, openai_sys_prompt):
    for i in range(number_of_iterations):
        openai_suggestion = request_to_openai(prompttracker,openai_sys_prompt)
        prompttracker.mistral_system_prompts.append(openai_suggestion)
        prompttracker.mistral_responses.append([])
        for j in range(len(prompttracker.target_input)):
            mistral_inputs = create_mistral_total_prompt(prompttracker.mistral_system_prompts[-1], prompttracker.target_input[j])
            output_from_mistral = local_llm_call(mistral_inputs)
            prompttracker.mistral_responses[-1].append(output_from_mistral)




def main_function(init_sys_prompt, openai_sys_prompt, input_evaluation_1, output_evaluation_1, input_evaluation_2, output_evaluation_2, counter):
    
    promptTracker.target_input.append(input_evaluation_1)
    promptTracker.target_input.append(input_evaluation_2)
    promptTracker.target_output.append(str(output_evaluation_1))
    promptTracker.target_output.append(str(output_evaluation_2))


    promptTracker.mistral_system_prompts.append(init_sys_prompt)

    promptTracker.mistral_responses.append([])
    for i in range(len(promptTracker.target_input)):

        initial_mistral_inputs = create_mistral_total_prompt(promptTracker.mistral_system_prompts[-1], promptTracker.target_input[i])

        output_from_mistral = local_llm_call(initial_mistral_inputs)
        promptTracker.mistral_responses[-1].append(output_from_mistral)
        
    #print(promptTracker.mistral_responses)
    refine_system_prompt_with_gpt4(counter, promptTracker, openai_sys_prompt)

    return promptTracker.mistral_system_prompts[-1]



input_evaluation_1_temp = "She said: 'today was supposed to be a day of celebration and joy in Kansas, instead it is another day where America has experience senselense gun violence' in response to what happened in Kansas near coca-cola branch"

output_evaluation_1_temp = ["Gun violence", "Coca-cola", "Kansas city"]

input_evaluation_2_temp = "I would say the best place to go for your honeymoon is Paris, but some say it's overrated"

output_evaluation_2_temp = ["Paris", "Honeymoon"]

init_sys_prompt_temp = """You are an AI designed to find a LIMITED list of GENERAL concepts associated with a given piece of text. The list size should NOT exceed 10. You Must use standardized words.

###
Here are some examples:


Text: "israel supporters attacks female palestine activist"
Concepts_List: ["Hate speech", "Palestine"]
###
"""



openai_sys_prompt_temp = """You are an AI assistant who is expert in creating promtps for LLMs. you job is to modify and enhance a prompt for a 7b mistral instruct model. The mistral model is supposed to receive an input text, and return a list of strings, entities, brand names, etc in that input text. This LLM is going to be used for .... The prompt to the mistral model can include some examples that lead the model's behavior. Mistral model performs constrained decoding, meaning that it only generates a list of strings.

A number of experiments have been done on different system prompts for mistral and the output. Those experiments which include tested system prompt, tested INPUTs TO MISTRAL, and the resulting outputs from Mistral are provided to you. Your job is to observe the experiments, and come up with a better system prompt for Mistral to achieve the expected output. you can provide some examples, or remove some examples in your suggested system prompt. Remember that total number of examples should be limited to 3, because it adds extra computation and we can't afford it. Note that the examples given in the system prompt of mistral should be enclosed by ### ###. Pay attention to the fact that, you are NOT allowed to use EVALUATION INPUT TO MISTRAL texts in your examples for your suggested mistral system prompt.
"""

def observe_prompts():
    instance_dict = asdict(promptTracker)
    instance_str = json.dumps(instance_dict, indent=4)
    return instance_str

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            init_sys_prompt = gr.Textbox(label="Initial Prompt", value=init_sys_prompt_temp)
            openai_sys_prompt = gr.Textbox(label="OpenAI Prompt", value=openai_sys_prompt_temp)
            input_evaluation_1 = gr.Textbox(label="input_evaluation_1", value=input_evaluation_1_temp)
            output_evaluation_1 = gr.Textbox(label="output_evaluation_1", value=output_evaluation_1_temp)
            input_evaluation_2 = gr.Textbox(label="input_evaluation_2", value=input_evaluation_2_temp)
            output_evaluation_2 = gr.Textbox(label="output_evaluation_2", value=output_evaluation_2_temp)
            counter = gr.Number(value=0, label="Counter Value")
            submit_button = gr.Button("Greet")

        with gr.Column():
            final_answer = gr.Textbox(label="openai final answer")
            new_button = gr.Button("observe")
            observe_prompttracker = gr.Textbox(label="prompt tracker")

        submit_button.click(fn=main_function, inputs=[init_sys_prompt, openai_sys_prompt, input_evaluation_1, output_evaluation_1, input_evaluation_2, output_evaluation_2, counter], outputs=[final_answer])
        new_button.click(fn=observe_prompts, inputs=[], outputs=[observe_prompttracker])


app.launch()