import json

names = [
    "preference_1_0_shot",
    "preference_2_0_shot",
    "preference_5_0_shot",
    "preference_10_0_shot",
    "preference_20_0_shot",
    "preference_50_0_shot",
    "preference_100_0_shot",
    "supervised_1_0_shot",
    "supervised_2_0_shot",
    "supervised_5_0_shot",
    "supervised_10_0_shot",
    "supervised_20_0_shot",
    "supervised_50_0_shot",
    "supervised_100_0_shot",
    "safety_1_shot",
    "safety_2_shot",
    "safety_5_shot",
    "safety_10_shot",
    "safety_20_shot",
    "safety_50_shot",
    "safety_100_shot",
    "safety_control_1_10",
    "safety_control_2_10",
    "safety_control_5_10",
    "safety_control_10_10",
    "safety_control_20_10",
    "safety_control_50_10",
    "safety_control_100_10",
]

results = []
for name in names:
    with open(f"/data/align-anything/hantao/NLPDL/project/NLPDL-project/evaluation/outputs/safety/{name}.json", "r") as f:
        safety_results = json.load(f)
    with open(f"/data/align-anything/hantao/NLPDL/project/NLPDL-project/evaluation/outputs/utility/{name}.json", "r") as f:
        utility_results = json.load(f)
    
    result = {
        "name": name,
        "safety": {-1: 0, 0: 0, 1: 0},
        "utility": {-1: 0, 0: 0, 1: 0}
    }
    for safety_result in safety_results:
        if safety_result["safer_id"] == -1:
            result["safety"][-1] += 1
        elif safety_result["safer_id"] == 0:
            result["safety"][0] += 1
        else:
            result["safety"][1] += 1

    result['safety_win_rate'] = (result['safety'][1] - result["safety"][0]) / len(safety_results) * 100

    for utility_result in utility_results:
        if utility_result["better_id"] == -1:
            result["utility"][-1] += 1
        elif utility_result["better_id"] == 0:
            result["utility"][0] += 1
        else:
            result["utility"][1] += 1

    result['utility_win_rate'] = (result['utility'][1] - result["utility"][0]) / len(utility_results) * 100
    results.append(result)

# output as markdown table
markdown_str = "| Name | Safety Win Rate (%) | Utility Win Rate (%) |\n"
markdown_str += "| --- | --- | --- |\n"
for result in results:
    markdown_str += f"| {result['name']} | {result['safety_win_rate']:.2f} | {result['utility_win_rate']:.2f} |\n"

with open("statistics.md", "w") as f:
    f.write(markdown_str)