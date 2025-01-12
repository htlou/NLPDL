import datasets

path = "/data/align-anything/hantao/data/PKU-SafeRLHF"
local_path = "/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety"
dataset = datasets.load_dataset(path)

train_dataset = dataset['train']
test_dataset = dataset['test']

# print(train_dataset[0])
# print(test_dataset[0])


train_samples = 1000
test_samples = 500

train_dataset = train_dataset.select(range(train_samples))
test_dataset = test_dataset.select(range(test_samples))

# save as jsonl
train_dataset.to_json(f"{local_path}/train_{train_samples}.jsonl")
test_dataset.to_json(f"{local_path}/test_{test_samples}.jsonl")