from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

from repe import repe_pipeline_registry
repe_pipeline_registry()

model_name_or_path = "/data/align-anything/hantao/models/alpaca-7b-reproduced"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side='left')

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)


import pickle
from utils import safety_function_dataset, plot_lat_scans, plot_detection_results
user_tag = "USER:"
assistant_tag = "ASSISTANT:"

samples_list = [1, 2, 5, 10, 20, 30, 50, 100, 200, 500]
for train_samples in samples_list:
    test_samples = 10
    
    if os.path.exists(f"outputs/safety_control_{train_samples}_{test_samples}.jsonl"):
        with open(f"outputs/safety_control_{train_samples}_{test_samples}.jsonl", "r") as f:
            tmp = [json.loads(line) for line in f]
        if len(tmp) == 500:
            print(f"Skipping {train_samples} samples, because it has already been processed")
            continue
        else:
            print(f"Processing {train_samples} samples, {len(tmp)} samples already processed")
    

    print(f"Loading dataset for {train_samples} samples")
    if os.path.exists(f"outputs/dataset_origin_{train_samples}_{test_samples}.pkl"):
        with open(f"outputs/dataset_origin_{train_samples}_{test_samples}.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        data_path = "/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety"
        dataset = safety_function_dataset(data_path, tokenizer, user_tag, assistant_tag, train_samples=train_samples, test_samples=test_samples, split="train")

        import pickle
        with open(f"outputs/dataset_origin_{train_samples}_{test_samples}.pkl", "wb") as f:
            pickle.dump(dataset, f)

    # with open(f"outputs/dataset_origin_{train_samples}_{test_samples}.pkl", "rb") as f:
    #     dataset = pickle.load(f)


    print(f"Loading safety vector for {train_samples} samples")
    if os.path.exists(f"outputs/safety_vector_{train_samples}_{test_samples}.pkl"):
        with open(f"outputs/safety_vector_{train_samples}_{test_samples}.pkl", "rb") as f:
            honesty_rep_reader = pickle.load(f)
    else:
        honesty_rep_reader = rep_reading_pipeline.get_directions(
            dataset['train']['data'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            train_labels=dataset['train']['labels'], 
            direction_method=direction_method,
            batch_size=64,
            padding='max_length',
            truncation=True,
        )

        import pickle
        with open(f"outputs/safety_vector_{train_samples}_{test_samples}.pkl", "wb") as f:
            pickle.dump(honesty_rep_reader, f)
    # read the reader with pickle
    # import pickle
    # with open("honesty_rep_reader.pkl", "rb") as f:
    #     honesty_rep_reader = pickle.load(f)
    # with open(f"outputs/safety_vector_{train_samples}_{test_samples}.pkl", "rb") as f:
    #     safety_rep_reader = pickle.load(f)

    # S_tests = rep_reading_pipeline(
    #     dataset['test']['data'], 
    #     rep_token=rep_token, 
    #     hidden_layers=hidden_layers, 
    #     rep_reader=safety_rep_reader,
    #     batch_size=64)

    if os.path.exists(f"outputs/safety_control_raw_{train_samples}_{test_samples}.pkl"):
        with open(f"outputs/safety_control_raw_{train_samples}_{test_samples}.pkl", "rb") as f:
            control_outputs = pickle.load(f)
    else:
        layer_id = list(range(-10, -32, -1))
        block_name="decoder_block"
        control_method="reading_vec"

        rep_control_pipeline = pipeline(
            "rep-control", 
            model=model, 
            tokenizer=tokenizer, 
            layers=layer_id, 
            control_method=control_method)

        template = " USER: {prompt} ASSISTANT: "

        inputs = []
        original_inputs = []

        input_paths = [
            "/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety/test_500.jsonl",
        ]

        full_data = []
        for input_path in input_paths:
            with open(input_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    inputs.append(template.format(prompt=data['prompt']))
                    original_inputs.append(data['prompt'])
                    full_data.append(data)
        # coeff_list = [-1.5, -1.0, -0.7, -0.3, 0, 0.3, 0.7, 1.0, 1.5]
        coeff=0.5
        max_new_tokens=512

        # baseline_outputs = rep_control_pipeline(inputs, batch_size=32, max_new_tokens=max_new_tokens, do_sample=False)
        activations = {}
        for layer in layer_id:
            activations[layer] = torch.tensor(coeff * honesty_rep_reader.directions[layer] * honesty_rep_reader.direction_signs[layer]).to(model.device).half()

        print(f"Controling {train_samples} samples")
        control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=32, max_new_tokens=max_new_tokens, do_sample=False)
        import pickle
        with open(f"outputs/safety_control_raw_{train_samples}_{test_samples}.pkl", "wb") as f:
            pickle.dump(control_outputs, f)

    print(len(control_outputs))

    outputs = []
    for i in range(len(inputs)):
        # base = baseline_outputs[i][0]['generated_text'].replace(inputs[i], "")
        control = control_outputs[i][0]['generated_text'].replace(inputs[i], "")
        outputs.append({
            "prompt": full_data[i]['prompt'],
            # "baseline_output": base,
            "controlled_output": control
        })

    with open(f"outputs/safety_control_{train_samples}_{test_samples}.jsonl", "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

    torch.cuda.empty_cache()
