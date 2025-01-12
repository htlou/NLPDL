import json
import os

base_path = "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_0_shot.json"

new_paths = [
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_1_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_2_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_5_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_10_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_20_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_50_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/safety_100_shot.json",

    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_1_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_2_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_5_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_10_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_20_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_50_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/preference_100_0_shot.json",

    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_1_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_2_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_5_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_10_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_20_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_50_0_shot.json",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs/supervised_100_0_shot.json",

    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_1_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_2_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_5_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_10_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_20_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_30_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_50_10.jsonl",
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/experiments/safety/outputs/safety_control_100_10.jsonl"
]

base_data = json.load(open(base_path, "r"))
tf_base = []
for item in base_data:
    tf_base.append(
        {
            "prompt": item["prompt"],
            "response_0": item["generated"][0]["response"]
        }
    )

# build a dict using the prompt as the key
prompt_dict = {}
for item in tf_base:
    prompt_dict[item["prompt"]] = item

for new_path in new_paths:
    if not os.path.exists(new_path):
        print(f"file {new_path} not found, skipping")
        continue

    if new_path.endswith(".json"):
        new_data = json.load(open(new_path, "r"))
    else:
        new_data = []
        for line in open(new_path, "r"):
            new_data.append(json.loads(line))
    
    tf_new = []
    for item in new_data:
        base_item = prompt_dict.get(item["prompt"], None)
        if base_item is None:
            raise ValueError(f"prompt {item['prompt']} not found in base data")
        try:
            response1 = item['generated'][0]["response"]
        except:
            response1 = item['controlled_output']
        tf_new.append({
            "prompt": item["prompt"],
            "response_0": base_item["response_0"],
            "response_1": response1,
        })

    # get the filename without the path and extension
    filename = os.path.basename(new_path).split(".")[0]
    with open(f"/data/align-anything/hantao/NLPDL/project/NLPDL-project/evaluation/data/{filename}.json", "w") as f:
        json.dump(tf_new, f, indent=4)