import json
import re

path = "outputs/gsm8k_reflexion_3iterations.json"

with open(path, "r") as f:
    data = json.load(f)

num = 0
correct = 0

for item in data:
    # 提取 ground truth 中 #### 后的数字
    ground_truth_match = re.search(r'####\s*(\d+)', item["ground_truth"])
    # 提取 final_answer 中的第一个数字
    if item["final_answer"] is not None:
        final_answer_match = re.search(r'\d+', item["final_answer"])
    else:
        final_answer_match = None

    # 确保匹配结果不为 None
    ground_truth = ground_truth_match.group(1) if ground_truth_match else None
    final_answer = final_answer_match.group(0) if final_answer_match else None

    print(f"Ground Truth: {ground_truth}, Final Answer: {final_answer}")
    
    if ground_truth and final_answer and final_answer == ground_truth:
        correct += 1

    num += 1

print(f"Accuracy: {correct}/{num}")