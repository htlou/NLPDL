import json
import random

path = "/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety/train_1000.jsonl"

# cut the dataset into (1 2 5 10 20 50 100) sizes, in preference and supervised separately

def transform_preference(dataset):
    outputs = []
    for data in dataset:
        if data['safer_response_id'] == 1:
            outputs.append({
                "prompt": data['prompt'],
                "chosen": data['response_1'],
                "rejected": data['response_0']
            })
        else:
            outputs.append({
                "prompt": data['prompt'],
                "chosen": data['response_0'],
                "rejected": data['response_1']
            })
    return outputs

def transform_supervised(dataset):
    outputs = []
    for data in dataset:
        if data['safer_response_id'] == 1:
            outputs.append({
                "prompt": data['prompt'],
                "response": data['response_1'],
            })
        else:
            outputs.append({
                "prompt": data['prompt'],
                "response": data['response_0'],
            })
    return outputs

full_dataset = []
with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        full_dataset.append(data)

sizes = [1, 2, 5, 10, 20, 50, 100]
random.shuffle(full_dataset)

# preference
for size in sizes:
    preference_dataset = transform_preference(full_dataset[:size])
    with open(f"preference_train_{size}.json", 'w') as f:
        json.dump(preference_dataset, f, indent=4)

# supervised
for size in sizes:
    supervised_dataset = transform_supervised(full_dataset[:size])
    with open(f"supervised_train_{size}.json", 'w') as f:
        json.dump(supervised_dataset, f, indent=4)

