from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets
import random
import os
import pandas as pd
from collections import Counter
import json

def get_dataset(dataset_name, sep_token, num_shots=3, verbose=False):
    '''
    dataset_name: str or list of str, the name(s) of the dataset(s)
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    
    datasets_list = []
    label_offset = 0
    label_map = {}
    for name in dataset_name:
        if name == 'restaurant_sup':
            ds, num_labels = load_restaurant_sup(sep_token)
        elif name == 'laptop_sup':
            ds, num_labels = load_laptop_sup(sep_token)
        elif name == 'acl_sup':
            ds, num_labels = load_acl_sup(sep_token)
        elif name == 'agnews_sup':
            ds, num_labels = load_agnews_sup(sep_token)
        elif "_fs" in name:
            sup_name = name.replace("_fs", "_sup")
            ds, label_map = get_dataset(sup_name, sep_token, verbose=True)
            num_labels = label_map[sup_name][1] - label_map[sup_name][0] + 1
            ds = create_few_shot_dataset(ds, num_examples_per_class=num_shots, num_labels=num_labels)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        # Re-label the labels to avoid overlaps
        ds = relabel_dataset(ds, label_offset)
        label_map[name] = (label_offset, label_offset + num_labels - 1)
        label_offset += num_labels

        datasets_list.append(ds)

    if len(datasets_list) == 1:
        dataset = datasets_list[0]
    else:
        # Concatenate datasets
        train_datasets = [ds['train'] for ds in datasets_list]
        test_datasets = [ds['test'] for ds in datasets_list]
        combined_train = concatenate_datasets(train_datasets)
        combined_test = concatenate_datasets(test_datasets)
        dataset = DatasetDict({'train': combined_train, 'test': combined_test})

    if verbose:
        return dataset, label_map
    return dataset

def relabel_dataset(dataset, label_offset):
    def add_offset(example):
        example['label'] += label_offset
        return example
    dataset = dataset.map(add_offset)
    # for example in dataset:
    #     example['label'] += label_offset

    return dataset

def read_file(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        return data
    else:
        raise ValueError(f"Unknown file type: {file_path}")

def load_restaurant_sup(sep_token):

    train_file = './data/SemEval14-res/train.json'
    test_file = './data/SemEval14-res/test.json'

    polarity_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    num_labels = len(polarity_map)

    # Load train data
    train_df = read_file(train_file)
    train_texts = [train_df[key]['term'] + sep_token + train_df[key]['sentence'] for key in train_df.keys()]
    train_labels = [polarity_map[train_df[key]['polarity']] for key in train_df.keys()]

    # Load test data
    test_df = read_file(test_file)
    test_texts = [test_df[key]['term'] + sep_token + test_df[key]['sentence'] for key in test_df.keys()]
    test_labels = [polarity_map[test_df[key]['polarity']] for key in test_df.keys()]

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    return dataset, num_labels

def load_laptop_sup(sep_token):
    # Similar to load_restaurant_sup, adjust file paths and data accordingly
    train_file = './data/SemEval14-laptop/train.json'
    test_file = './data/SemEval14-laptop/test.json'

    polarity_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    num_labels = len(polarity_map)

    # Load train data
    train_df = read_file(train_file)
    train_texts = [train_df[key]['term'] + sep_token + train_df[key]['sentence'] for key in train_df.keys()]
    train_labels = [polarity_map[train_df[key]['polarity']] for key in train_df.keys()]

    # Load test data
    test_df = read_file(test_file)
    test_texts = [test_df[key]['term'] + sep_token + test_df[key]['sentence'] for key in test_df.keys()]
    test_labels = [polarity_map[test_df[key]['polarity']] for key in test_df.keys()]

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    return dataset, num_labels

def load_acl_sup(sep_token):
    # Load the ACL-ARC dataset
    train_file = './data/ACL-ARC/train.jsonl'
    test_file = './data/ACL-ARC/test.jsonl'

    # Assuming labels are integers starting from 0
    label_map = {'Uses': 0, 'Future': 1, 'Extends': 2, 'Motivation': 3, 'CompareOrContrast': 4, 'Background': 5} 
    # Load train data
    train_df = read_file(train_file)
    train_texts = [train_df[i]['text'] for i in range(len(train_df))]
    train_labels = [label_map[train_df[i]['label']] for i in range(len(train_df))]
    num_labels = len(set(train_labels))

    # Load test data
    test_df = read_file(test_file)
    test_texts = [test_df[i]['text'] for i in range(len(test_df))]
    test_labels = [label_map[test_df[i]['label']] for i in range(len(test_df))]

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    return dataset, num_labels

def load_agnews_sup(sep_token):
    # Load the test split of ag_news dataset
    dataset = load_dataset('ag_news', split='test')

    # Randomly split the dataset into training and test sets with ratio 9:1
    dataset = dataset.train_test_split(test_size=0.1, seed=2022)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    num_labels = len(set(train_dataset['label']))

    return dataset, num_labels

def create_few_shot_dataset(sup_dataset, num_examples_per_class, num_labels, seed=2022, sep_token='<sep>'):
    from datasets import DatasetDict, Dataset
    import random

    random.seed(seed)

    train_dataset = sup_dataset['train']
    test_dataset = sup_dataset['test']

    # For each label, sample num_examples_per_class examples
    label_to_indices = {}
    for idx, label in enumerate(train_dataset['label']):
        label_to_indices.setdefault(label, []).append(idx)

    sampled_indices = []
    for label in range(num_labels):
        indices = label_to_indices.get(label, [])
        if len(indices) >= num_examples_per_class:
            sampled = random.sample(indices, num_examples_per_class)
        else:
            sampled = indices  # Use all available if not enough
        sampled_indices.extend(sampled)

    # Create new train dataset with sampled indices
    sampled_train_dataset = train_dataset.select(sampled_indices)

    # Keep the test dataset as is
    fs_dataset = DatasetDict({'train': sampled_train_dataset, 'test': test_dataset})

    return fs_dataset

def __main__():
    # zero_shot_dataset_list = ['agnews_sup','restaurant_sup','laptop_sup','acl_sup']
    # few_shot_dataset_list = ['restaurant_fs', 'laptop_fs', 'acl_fs', 'agnews_fs']
    sep_token = '<sep>'
    # for dataset_name in zero_shot_dataset_list:
    #     dataset = get_dataset(dataset_name, sep_token)
    #     print(f"The first example of {dataset_name} dataset is: {dataset['train'][0]}")
    #     # print(f"The number of labels of {dataset_name} dataset is: {dataset['train'].features['label'].num_classes}")
    #     print(f"The number of training examples of {dataset_name} dataset is: {len(dataset['train'])}")
    #     labels = [example['label'] for example in dataset['train']]
    #     label_distribution = Counter(labels)
    #     print(f"The label distribution of {dataset_name} dataset is: {label_distribution}")

    # for dataset_name in few_shot_dataset_list:
    #     dataset = get_dataset(dataset_name, sep_token, num_shots=5)
    #     print(f"The first example of {dataset_name} dataset is: {dataset['train'][0]}")
    #     # print(f"The number of labels of {dataset_name} dataset is: {dataset['train'].features['label'].num_classes}")
    #     print(f"The number of training examples of {dataset_name} dataset is: {len(dataset['train'])}")

    #     labels = [example['label'] for example in dataset['train']]
    #     label_distribution = Counter(labels)
    #     print(f"The label distribution of {dataset_name} dataset is: {label_distribution}")
    
    multiple_dataset_names = ['restaurant_fs', 'laptop_fs', 'acl_fs']
    dataset = get_dataset(multiple_dataset_names, sep_token, num_shots=5)
    print(f"The first example of {multiple_dataset_names} dataset is: {dataset['train'][0]}")
    # print(f"The number of labels of {multiple_dataset_names} dataset is: {dataset['train'].features['label'].num_classes}")
    print(f"The number of training examples of {multiple_dataset_names} dataset is: {len(dataset['train'])}")
    labels = [example['label'] for example in dataset['train']]
    label_distribution = Counter(labels)
    print(f"The label distribution of {multiple_dataset_names} dataset is: {label_distribution}")

if __name__ == '__main__':
    __main__()