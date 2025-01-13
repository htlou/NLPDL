# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__  import annotations
import json
import os
import argparse

import sys
from tqdm.auto import tqdm
from vllm import SamplingParams,LLM

PROMPT_USER: str = 'USER: {prompt}'
PROMPT_ASSISTANT: str = 'ASSISTANT: {response}'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_USER + PROMPT_ASSISTANT
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import requests
import datasets
import io
import random

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )
    # Model
    parser.add_argument(
       '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
    )
    parser.add_argument(
        '--input_path',
        type=str,
        help='the path of the input json file',
    )
    parser.add_argument(
        '--few_shot_path',
        type=str,
        help='the path of the few shot json file',
    )
    parser.add_argument(
        '--few_shot_num',
        type=int,
        default=0,
        help='the number of few shot',
    )
    parser.add_argument(
        '--output_name',
        type=str,
        help='the name of the output json file',
        default=None,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Where to store the eval output.',
    )
    return parser.parse_args()


def generate_answer_by_vllm(problems: list[str], model_name_or_path:str) ->list[str]:
    samplingparams = SamplingParams(
        temperature = 0.05,
        repetition_penalty = 1.0,
        max_tokens = 4096,
        n=1,
    )
    
    llm = LLM(model=model_name_or_path, 
              gpu_memory_utilization=0.9, 
              swap_space=16, 
              trust_remote_code=True, 
              tensor_parallel_size=8)
    model_name = model_name_or_path.split('/')[-1]
    outputs = llm.generate(problems, samplingparams)
    answers = []
    
    for output, entry in tqdm(zip(outputs, problems)) :
        items = []
        for i in range(len(output.outputs)):
            item = {
                'from': model_name,
                'response': output.outputs[i].text.strip()
            }
            items.append(item)
        answers.append(items)
    
    return answers

def main() -> None:
    args = parse_arguments()
    input_path = args.input_path

    if args.few_shot_path is not None:
        # load from jsonl
        few_shot_data = datasets.load_dataset(args.few_shot_path, split='train')
        
        few_shot_data = few_shot_data.to_list()

        random.shuffle(few_shot_data)
        few_shot_data = few_shot_data[:args.few_shot_num]
    
    else:
        few_shot_data = []
    
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raw_data = datasets.load_dataset(input_path, split='test')
        data = []
        for item in raw_data:
            data.append({
                'prompt': item.get('question', item.get('prompt')),
            })
    
    print(f"input_path: {input_path}")
 
    problems = []
    few_shot_prompt = ""
    if args.few_shot_num > 0:
        for i in range(len(few_shot_data)):
            if few_shot_data[i]['safer_response_id'] == 1:
                few_shot_prompt += PROMPT_INPUT.format(prompt=few_shot_data[i]['prompt'], response=few_shot_data[i]['response_1'])
            else:
                few_shot_prompt += PROMPT_INPUT.format(prompt=few_shot_data[i]['prompt'], response=few_shot_data[i]['response_0'])

    for idx in range(len(data)):
        raw_prompt = PROMPT_INPUT.format(prompt=data[idx]['prompt'], response="")
        problem = {
            'prompt': few_shot_prompt + raw_prompt,
        }
        problems.append(problem)
        
    answers = generate_answer_by_vllm(problems, args.model_name_or_path)
    final_answer = []
    for idx in range(len(answers)):
        item = answers[idx]
        final_answer.append(item)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, answer in enumerate(answers):
        data[idx]['generated'] = answer
    if args.output_name is None:
        args.output_name = f"generated_base_{args.model_name_or_path.split('/')[-1]}.json"
    else:
        args.output_name = f"{args.output_name}"
    output_file = os.path.join(args.output_dir, args.output_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
if __name__=='__main__':
    sys.exit(main())