
import argparse
import hashlib
import itertools
import json
import logging
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Union
import io
from PIL import Image

import openai
import ray
import tqdm
# from tqdm import tqdm

import urllib3
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from web_utils_openai import *

# from prompts import *
# from translate_en_ch.system_prompt import SYSTEM_PROMPT,USER_PROMPT
from config.system_prompt import SAFETY_SCORE_SYSTEM_PROMPT,SAFETY_SCORE_USER_PROMPT
from config.system_prompt import UTILITY_SCORE_SYSTEM_PROMPT,UTILITY_SCORE_USER_PROMPT
from config.system_prompt import HELPFUL_SCORE_SYSTEM_PROMPT,HELPFUL_SCORE_USER_PROMPT
from config.system_prompt import REASONING_SCORE_SYSTEM_PROMPT,REASONING_SCORE_USER_PROMPT
from config.system_prompt import IMAGE_RECOGNITION_SYSTEM_PROMPT,IMAGE_RECOGNITION_USER_PROMPT
from config.system_prompt import IMAGE_JUDGE_SYSTEM_PROMPT, IMAGE_JUDGE_USER_PROMPT
from config.system_prompt import IMAGE_SCORE_SYSTEM_PROMPT, IMAGE_SCORE_USER_PROMPT
from config.system_prompt import INTERLEAVED_SCORE_SYSTEM_PROMPT, INTERLEAVED_SCORE_USER_PROMPT
from config.system_prompt import GSM8K_SCORE_SYSTEM_PROMPT, GSM8K_SCORE_USER_VANILLA_PROMPT, GSM8K_SCORE_USER_COT_PROMPT, GSM8K_SCORE_USER_FEW_SHOT_PROMPT, GSM8K_SCORE_USER_FEW_SHOT_PROMPT_2SHOT, GSM8K_SCORE_USER_COT_FEW_SHOT_PROMPT

HERE = Path(__file__).absolute().parent

DEFAULT_OPENAI_CHAT_COMPLETION_MODELS = [
    'deepseek-chat'
]
# DEFAULT_OPENAI_CHAT_COMPLETION_MODELS = [
#     'gpt-3.5-turbo'
# ]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((512, 512))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def split_text_by_image_tag(text, tag='<image>'):
    # 利用正则表达式查找标记，并确保标记在分割后的字符串中保留
    # 正则表达式说明：
    # (?<=...) 是一个正向后查找，确保所匹配的位置前有特定字符串，这里是 tag
    # (?=...) 是一个正向前查找，确保所匹配的位置后有特定字符串，这里是 tag 或字符串末尾
    pattern = f'(?<={re.escape(tag)})|(?={re.escape(tag)}$)'
    # 使用 re.split 分割字符串，但不移除分割符
    parts = re.split(pattern, text)
    # 移除空字符串，可能由于连续的标记产生
    parts = [part for part in parts if part]
    return parts

def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid

def hash_checker(result: dict):
    return (result['safer_id'] is not None) or result['better_id'] is not None

def string_to_seed(s):
    """将字符串转换为整数种子"""
    # 使用hashlib生成字符串的哈希值
    hash_value = hashlib.sha256(s.encode()).hexdigest()
    # 将哈希值（16进制字符串）转换为整数
    seed = int(hash_value, 16)
    return seed

def fixed_random(s):
    """根据给定的字符串s生成固定范围[a, b]内的随机整数"""
    seed = string_to_seed(s)
    random.seed(seed)
    return random.choice([0,1])

def post_process(response, type: str):
    order = response['order']
    #(.*?) 表示非贪婪模式的捕获组，匹配任意字符（除了换行符 \n）零次或多次
    if(type == "safety"):
        pattern = r'More harmful: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'],'safer_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['safer_id'] = 1 - order
            elif match == 'responseb':
                result['safer_id'] = 0 + order
            elif match == 'equal':
                result['safer_id'] = -1
        return result
    elif(type == "utility"):
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'better_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif(type == "empathetic"):
        pattern = r'More empathetic: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'better_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif(type == "reasoning"):
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'better_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif "gsm8k" in type:
        # 查找最后一个 #### 后的内容
        pattern = r'####(.*)'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        gt_answer = response['transformed_input']['answer']
        gt_matches = re.findall(pattern, gt_answer, re.IGNORECASE)

        # 初始化结果字典
        result = {'qac': response['transformed_input'], 'Acc': None, 'reasoning': response['output']}

        # 提取数字部分的正则表达式
        number_pattern = r'\d+'

        if matches and gt_matches:
            # 提取最后一个匹配中的数字
            match_numbers = re.findall(number_pattern, matches[-1])
            gt_match_numbers = re.findall(number_pattern, gt_matches[-1])

            # 比较数字列表
            if match_numbers == gt_match_numbers:
                result['Acc'] = 1
            else:
                result['Acc'] = 0

        return result
    elif(type == "image-recognition"):
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'better_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif(type == "image-judge"):
        pattern = r'\[\[OUTPUT\]\](.*)'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'output': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            result['output'] = match
        return result
    elif type == "image-compare":
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'better_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'image-1':
                result['better_id'] = 0 + order
            elif match == 'image-2':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif type == "interleaved-compare":
        pattern = r'(?:\*\*)?Better(?:\*\*)?: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'], 'better_id': None, 'reasoning': response['output']}
        if len(matches) >= 1:
            match = matches[0].lower()
            if match == 'output1' or match == 'output-1':
                result['better_id'] = 0 + order
            elif match == 'output2' or match == 'output-2':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    else:
        raise RuntimeError("not implemented type")




@ray.remote(num_cpus=1)
def request_openai(
    id: int,
    type: str,
    input: dict[str, str],
    openai_api_keys: list[(str, str)],
    openai_model: str,
    base_url: str | None = None,
    cache_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    # print(cache_dir)
    openai_api_keys = itertools.cycle(openai_api_keys)
    openai_api_keys = next(openai_api_keys)
    
    platform = openai_api_keys[1]
    openai_api_key = openai_api_keys[0]
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = input['system_prompt']
    user_prompt = input['user_prompt']
    sha256 = hashlib.sha256((system_prompt + user_prompt).encode('utf-8')).hexdigest()

    if cache_dir is not None:
        output_file = cache_dir / f'{sha256}.json'
        # print(output_file)
        if output_file.exists():
            with output_file.open(mode='rt', encoding='utf-8') as f:
                try:
                    result = json.load(f)
                    return id, result
                except json.JSONDecodeError:
                    output_file = None
    else:
        output_file = None
    
    
    # print(output_file)

    if (type == "image-recognition"):
        # print(input)
        image_url = input['transformed_input']['image_url']
        #print(image_url)
        messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 
         'content': [
             {
                 "type": "text",
                 "text": user_prompt
             },
             {
                 "type": "image_url",
                 "image_url": {
                    "url": f"data:image/jpeg;base64,{image_url}"
                },
             }
         ]
        },
      ]
        #print(messages)
    elif type == "image-judge":
        # example_url = input['transformed_input']['example_url']
        image = input['transformed_input']['image']
        print(input['transformed_input']['image_url'])
        print(len(image))
        messages = [
        {
            'role': 'system', 
            'content': system_prompt,
            # 'content': [
            #     # {
            #     #     "type": "image_url",
            #     #     "image_url": {
            #     #         "url": f"data:image/jpeg;base64,{example_url}"
            #     #     },
            #     # },
            #     {
            #         "text": system_prompt
            #     },
            # ]
            
        },
        {
            'role': 'user', 
            'content': [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt
                },
            ]
        },
      ]
    elif type == "image-compare":
        image_a = input['transformed_input']['image_a']
        image_b = input['transformed_input']['image_b']
        url_a = input['transformed_input']['url_a']
        url_b = input['transformed_input']['url_b']
        
        
        file_type_mapping = {
            'png': 'png',
            'jpeg': 'jpg',
            'jpg': 'jpg',
            'webp': 'webp'
        }
        
        try:
            _, ext_a = os.path.splitext(url_a)
            type_a = file_type_mapping.get(ext_a.lower(), 'png')
        except:
            type_a = 'png'
        try:
            _, ext_b = os.path.splitext(url_b)
            type_b = file_type_mapping.get(ext_b.lower(), 'png')
        except:
            type_b = 'png'
        
        
        messages = [
        {
            'role': 'system', 
            'content': system_prompt,
        },
        {
            'role': 'user', 
            'content': [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{type_a};base64,{image_a}"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{type_b};base64,{image_b}"
                    },
                },
            ]
        },
      ]
    elif type == "interleaved-compare":
        
        images = input['transformed_input']['images']
        split_user_prompt = split_text_by_image_tag(user_prompt)
        
        messages = []
        messages.append(
            {
            'role': 'system', 
            'content': system_prompt,   
            },
        )
        user_message = {
            'role': 'user', 
            'content': [],
        }
        
        assert len(images) == len(split_user_prompt) - 1, "image tokens in input should be the same length as the images!"
        
        for i in range(len(images)):
            text_piece = split_user_prompt[i]
            user_message['content'].append(
                {
                    "type": "text",
                    "text": text_piece,
                },
            )
            
            image_piece = images[i]
            user_message['content'].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_piece['type']};base64,{image_piece['content']}"
                    },
                },
            )
            
        user_message['content'].append(
                {
                    "type": "text",
                    "text": split_user_prompt[-1],
                },
            )
        
        messages.append(user_message)
        
    else:
        #print("Not working")
        messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    #print("work")
    result = input.copy()

    result.update(
        request_openai_noexcept(
            messages=messages,
            openai_api_keys=openai_api_key,
            openai_model=openai_model,
            base_url=base_url,
        ),
    )
    
    result['sha256'] = sha256
    # print(result)
    # result['transformed_input']['image'] = None
    # result['message'] = None
    # print(result)
    if output_file is not None:
        with output_file.open(mode='wt', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return id, result


def batch_request_openai(
    type: str,
    inputs: list[dict[str, Any]],
    openai_api_keys: list[str],
    openai_models: list[str],
    base_url: str | None = None,
    num_workers: int = 8,
    cache_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    openai_api_keys = sorted(set(openai_api_keys))
    openai_models = sorted(set(openai_models))
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    pending = deque(enumerate(inputs))
    not_ready = []
    results = [None for _ in range(len(pending))]
    openai_api_keys_cycle = itertools.cycle(
        [openai_api_keys[i:] + openai_api_keys[:i] for i in range(len(openai_api_keys))],
    )
    with tqdm.tqdm(total=len(pending)) as pbar:
        while len(not_ready) > 0 or len(pending) > 0:
            while len(not_ready) < num_workers and len(pending) > 0:
                idx, input = pending.popleft()
                current_key=next(openai_api_keys_cycle)
                # print(type)
                not_ready.append(
                    request_openai.remote(
                        idx,
                        type,
                        input,
                        openai_api_keys=current_key,
                        openai_model=random.choice(openai_models),  # noqa: S311
                        base_url=base_url,
                        cache_dir=cache_dir,
                    ),
                )
                

            ready, not_ready = ray.wait(not_ready, timeout=1)
            for idx, result in ray.get(ready):
                results[idx] = result
            pbar.update(len(ready))

    return results


def get_openai_api_keys(
    openai_api_keys: list[str],
    openai_api_key_file: Path | str | None,
) -> list[str]:
    openai_api_keys = list(openai_api_keys or [])

    if openai_api_key_file is not None:
        openai_api_key_file = Path(openai_api_key_file).expanduser().absolute()
        with openai_api_key_file.open(mode='rt', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'#.*', '', line).strip()
                parts = tuple(line.split(','))
                if not line:
                    continue
                if not line.startswith('sk-'):
                    raise ValueError(f'Invalid OpenAI API key: {line}')
                openai_api_keys.append(parts)

    openai_api_keys = list(dict.fromkeys(openai_api_keys))

    if len(openai_api_keys) == 0:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is not None:
            openai_api_keys.append(openai_api_key)
        else:
            raise ValueError('No OpenAI API key provided.')

    for i, [openai_api_key, platform] in enumerate(openai_api_keys, start=1):
        if not openai_api_key.startswith('sk-'):
            raise ValueError(f'Invalid OpenAI API key: {openai_api_key}')
        print(f'{platform} API key #{i}: {openai_api_key}')

    return openai_api_keys

def get_annotator_response_b_prompt(data, type):
    # restriction_list = RestrictionList #### 第一步所有restrictions都是violation
    
    if(type == "safety"):
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        user_prompt = SAFETY_SCORE_USER_PROMPT.format(
            prompt=question,
            responseA=response_a,
            responseB=response_b,
        )
        return SAFETY_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "utility"):
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        user_prompt = UTILITY_SCORE_USER_PROMPT.format(
            prompt=question,
            responseA=response_a,
            responseB=response_b,
        )
        return UTILITY_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "empathetic"):
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        user_prompt = HELPFUL_SCORE_USER_PROMPT.format(
            prompt=question,
            responseA=response_a,
            responseB=response_b,
        )
        return HELPFUL_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "reasoning"):
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        ground_truth = str(data['target'])

        user_prompt = REASONING_SCORE_USER_PROMPT.format(
            question=question,
            gt = ground_truth,
            response_a=response_a,
            response_b=response_b,
        )
        return REASONING_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "gsm8k"):
        question = str(data['question'])
        answer = str(data['answer'])
        user_prompt = GSM8K_SCORE_USER_VANILLA_PROMPT.format(
            question=question,
        )
        return GSM8K_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "gsm8k_cot"):
        question = str(data['question'])
        answer = str(data['answer'])
        user_prompt = GSM8K_SCORE_USER_COT_PROMPT.format(
            question=question,
        )
        return GSM8K_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "gsm8k_few_shot"):
        question = str(data['question'])
        answer = str(data['answer'])
        user_prompt = GSM8K_SCORE_USER_FEW_SHOT_PROMPT.format(
            question=question,
        )
        return GSM8K_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "gsm8k_few_shot_2shot"):
        question = str(data['question'])
        answer = str(data['answer'])
        user_prompt = GSM8K_SCORE_USER_FEW_SHOT_PROMPT_2SHOT.format(
            question=question,
        )
        return GSM8K_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "gsm8k_cot_few_shot"):
        question = str(data['question'])
        answer = str(data['answer'])
        user_prompt = GSM8K_SCORE_USER_COT_FEW_SHOT_PROMPT.format(
            question=question,
        )
        return GSM8K_SCORE_SYSTEM_PROMPT, user_prompt
    elif(type == "image-recognition"):
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])

        user_prompt = IMAGE_RECOGNITION_USER_PROMPT.format(
            question=question,
            response_a = response_a,
            response_b = response_b,
        )
        return IMAGE_RECOGNITION_SYSTEM_PROMPT, user_prompt
    elif type == "image-judge":
        question = str(data['question'])
        user_prompt = IMAGE_JUDGE_USER_PROMPT.format(
            text = question,
        )
        return IMAGE_JUDGE_SYSTEM_PROMPT, user_prompt
    
    elif type == "image-compare":
        question = str(data.get('prompt', data.get('question')))
        user_prompt = IMAGE_SCORE_USER_PROMPT.format(
            prompt = question,
        )
        return IMAGE_SCORE_SYSTEM_PROMPT, user_prompt
    elif type == "interleaved-compare":
        question = str(data.get('prompt', data.get('question')))
        output1 = str(data.get('response_a', data.get('output1')))
        output2 = str(data.get('response_b', data.get('output2')))
        user_prompt = INTERLEAVED_SCORE_USER_PROMPT.format(
            prompt = question,
            output1 = output1,
            output2 = output2,
        )
        return INTERLEAVED_SCORE_SYSTEM_PROMPT, user_prompt
        
    else:
        raise RuntimeError("not implemented type")

def transform_data(data, type, dictionary=None, root_url=None):
    # order = fixed_random(data.get('prompt', data.get('question')))
    order = 0
    # 0 原序 1 反序
    # 新的字典
    new_data = {}
    if type == "reasoning":
        new_data["target"] = data['target']
        new_data["question"] = data.get('prompt', data.get('question'))
        if order == 0:
            new_data["response_a"] = data.get('original_response_a', data.get('response_a'))
            new_data["response_b"] = data.get('response_b_response_a', data.get('response_b'))
        else:
            new_data["response_a"] = data.get('response_b_response_a', data.get('response_b'))
            new_data["response_b"] = data.get('original_response_a', data.get('response_a'))

        new_data['order'] = order
        return new_data, order

    if type=="gsm8k" or "gsm8k" in type:
        return data, 0

    if type == "image-recognition":
        new_data["question"] = data.get('prompt', data.get('question'))
        if order == 0:
            new_data["response_a"] = data.get('original_response_a', data.get('response_a'))
            new_data["response_b"] = data.get('response_b_response_a', data.get('response_b'))
        else:
            new_data["response_a"] = data.get('response_b_response_a', data.get('response_b'))
            new_data["response_b"] = data.get('original_response_a', data.get('response_a'))

        new_data['order'] = order
        new_data['image_url'] = data.get('image_url')
        return new_data, order
    
    if type == "image-judge":
        
        file_name = data
        
        new_data["question"] = dictionary['p']
        new_data['order'] = order
        # new_data['image_url'] = data.get('image_url')
        # print(root_url, file_name)
        new_data['image_url'] = Path.joinpath(root_url, file_name)
        new_data['image'] = encode_image(new_data['image_url'])

        
        # new_data['example_url'] = encode_image('/aifs4su/yaodong/projects/hantao/gpt4_eval/example.png')
        return new_data, order
    
    if type == "image-compare":
        new_data['question'] = data.get('prompt', data.get('question'))
        try:
            path_a = data['images'][1]
            image_a = encode_image(path_a)
        except:
            image_a = None
        
        try:
            path_b = data['images'][0]
            image_b = encode_image(path_b)
        except:
            image_b = None
            
        if order == 0:
            new_data['image_a'] = image_a
            new_data['image_b'] = image_b
            new_data['url_a'] = path_a
            new_data['url_b'] = path_b
        else:
            new_data['image_a'] = image_b
            new_data['image_b'] = image_a
            new_data['url_a'] = path_b
            new_data['url_b'] = path_a
            
        return new_data, order

    if type == "interleaved-compare":
        new_data['question'] = data.get('prompt', data.get('question'))
        new_data['output1'] = data.get('response_a', data.get('output1'))
        new_data['output2'] = data.get('response_b', data.get('output2'))
        
        file_type_mapping = {
            'png': 'png',
            'jpeg': 'jpeg',
            'jpg': 'jpeg',
            'webp': 'webp'
        }
        
        images = []
        for image_url in data['images']:
            try:
                _, ext = os.path.splitext(image_url)
                image_type = file_type_mapping.get(ext.lower(), 'png')
            except:
                image_type = 'png'
            
            try:
                image = encode_image(image_url)
            except:
                image = None
                
            formatted_image = {
                "type": image_type,
                "content": image,
            }
            images.append(formatted_image)
        
        new_data['images'] = images
        
        return new_data, order
    
    # 获取category中为True的键作为violation
    if ('category' not in data) or (not data['category']):
        violation = []
    else:
        violation = data['category']
    new_data["violation"] = violation
    new_data["question"] = data.get('prompt', data.get('question'))
    if order == 0:
        new_data["response_a"] = data.get('original_response_a', data.get('response_a'))
        new_data["response_b"] = data.get('response_b_response_a', data.get('response_b'))
    else:
        new_data["response_a"] = data.get('response_b_response_a', data.get('response_b'))
        new_data["response_b"] = data.get('original_response_a', data.get('response_a'))
    new_data['order'] = order
    return new_data, order

def prepare_inputs(input_file: Path | str, shuffle: bool = False, type: str = "image-recognition", platform : str = "openai") -> list[Any]:
    input_file = Path(input_file).expanduser().absolute()
    # print(input_file)
    input_root = input_file.parent

    if input_file.suffix.lower() == '.json':
        with input_file.open(mode='rt', encoding='utf-8') as f:
            raw_inputs = json.load(f)
    elif input_file.suffix.lower() == '.jsonl':
        with input_file.open(mode='rt', encoding='utf-8') as f:
            raw_inputs = [json.loads(line) for line in f]

    inputs = []
    i=0
    # print(raw_inputs)
    for raw_input in raw_inputs:
        # print(raw_input)
        data, order = transform_data(raw_input, type)
        # print(data)
        # print(data)
        system_prompt, user_prompt = get_annotator_response_b_prompt(data, type)
        inputs.append(
                {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'transformed_input': data,
                    'input_file': str(input_file),
                    'order': order
                },
            )
        i+=1

    if shuffle:
        random.shuffle(inputs)
    return inputs