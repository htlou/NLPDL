import pandas as pd
import json
import numpy as np
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from datasets import load_dataset

def safety_function_dataset(
        data_path: str, 
        tokenizer: PreTrainedTokenizer, 
        user_tag: str = "", 
        assistant_tag: str = "", 
        seed: int = 42,
        train_samples: int = 1000,
        test_samples: int = 500,
        split: str = "train"
    ) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    # with open(data_path, 'r') as file:
    #     data = json.load(file)
    # load from jsonl
    data = load_dataset(data_path, split=split)

    # template_str = "Pretend you're {type} person making statements about the world."
    template = "{user_tag}{input}{assistant_tag}{answer}"
    # data = random.sample(data, 300)
    
    # Process statements

    train_original = random.sample(data.to_list(), train_samples)
    test_original = random.sample(data.to_list(), test_samples)

    def get_good_and_bad(data):
        good_statements = []
        bad_statements = []
        for piece in tqdm(data, desc="Processing data"):
            question = piece['prompt']
            if piece['safer_response_id'] == 1:
                chosen = piece['response_1']
                rejected = piece['response_0']
            else:
                chosen = piece['response_0']
                rejected = piece['response_1']

            good = template.format(user_tag=user_tag, input=question, assistant_tag=assistant_tag, answer=chosen)
            bad = template.format(user_tag=user_tag, input=question, assistant_tag=assistant_tag, answer=rejected)
            pub = template.format(user_tag=user_tag, input=question, assistant_tag=assistant_tag, answer='')

            good_tokens = tokenizer.tokenize(good)
            bad_tokens = tokenizer.tokenize(bad)
            pub_tokens = tokenizer.tokenize(pub)
            # good_statements.append(good_tokens)
            # bad_statements.append(bad_tokens)
            # good_statements.append(good)
            # bad_statements.append(bad)
            # 获取最大长度，以便知道循环的终止条件
            min_len = min(len(good_tokens), len(bad_tokens))
            max_len = max(len(good_tokens), len(bad_tokens))

            # 从pub_tokens的长度开始，到max_len结束
            for idx in range(min(len(pub_tokens)+5,min_len), min(min_len + 20,max_len)):
                if idx < len(good_tokens):
                    truncated_good_tokens = good_tokens[:idx]
                else:
                    truncated_good_tokens = good_tokens

                if idx < len(bad_tokens):
                    truncated_bad_tokens = bad_tokens[:idx]
                else:
                    truncated_bad_tokens = bad_tokens

                good_statement = tokenizer.convert_tokens_to_string(truncated_good_tokens)
                bad_statement = tokenizer.convert_tokens_to_string(truncated_bad_tokens)
                
                good_statements.append(good_statement)
                bad_statements.append(bad_statement)

        return good_statements, bad_statements

    # Create training data
    train_good, train_bad = get_good_and_bad(train_original)
    test_good, test_bad = get_good_and_bad(test_original)
    train_data = [[good, bad] for good, bad in zip(train_good, train_bad)]
    print(f"Combined data has a length of {len(train_data)}")

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    # print([type(item) for item in train_data])  # 检查类型
    # print([len(item) for item in train_data if isinstance(item, list)])  # 如果是列表，检查长度

    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[good, bad] for good, bad in zip(test_good[:-1], test_bad[1:])]).flatten()
    test_data = reshaped_data.tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":"):

    cmap=LinearSegmentedColormap.from_list('rg',["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Define words and their colors
    words = [token.replace('▁', ' ') for token in input_ids]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(12.8, 10), dpi=200)

    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 10)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 8
    y_pad = 0.3
    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.3
    word_width = 0

    iter = 0

    selected_concepts = ["correction"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean # get rid of outliers
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD # change this for threshold
            rep_scores = rep_scores / np.std(rep_scores[5:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # ofs = 0
        # rep_scores = np.array([rep_scores[max(0, i-ofs):min(len(rep_scores), i+ofs)].mean() for i in range(len(rep_scores))]) # add smoothing
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)


        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False
            
        for word, score in zip(words[5:], rep_scores[5:]):

            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            
            color = colormap(norm(score))

            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 3

            # Compute the width of the current word
            text = ax.text(x, y, word, fontsize=13)
            word_width = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
            word_height = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).height

            # Remove the previous text
            if iter:
                text.remove()

            # Add the text with background color
            text = ax.text(x, y + y_pad * (iter + 1), word, color='white', alpha=0,
                        bbox=dict(facecolor=color, edgecolor=color, alpha=0.8, boxstyle=f'round,pad=0', linewidth=0),
                        fontsize=13)
            
            # Update the x position for the next word
            x += word_width + 0.1
        
        iter += 1


def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice, fig_name):
    for rep, scores in rep_reader_scores_dict.items():

        start_tok = input_ids.index('▁|')

        print(start_tok, len(input_ids), (len(input_ids)-start_tok)//50, np.array(scores).shape)
        # print("Start token index:", start_tok)
        # print("Slice start index:", start_tok-20)
        # print("Slice end index:", start_tok+20)
        # standardized_scores = np.array(scores)[0:len(input_ids):(len(input_ids)-0)//50,layer_slice]
        standardized_scores = np.array(scores)[start_tok+25:start_tok+45,layer_slice]
        # standardized_scores = np.array(scores)[0:start_tok,layer_slice]
        print(standardized_scores.shape)

        # bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 3
        print(bound)
        print(f"Ideal Bound: {np.mean(standardized_scores) + np.std(standardized_scores)}")

        # standardized_scores = np.array(scores)
        
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)
        
        cmap = 'coolwarm'

        fig, ax = plt.subplots(figsize=(8, 6.4), dpi=1000)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)
        ax.tick_params(axis='y', rotation=0)

        ax.set_xlabel("Token Position", fontsize=25)#, fontsize=20)
        if fig_name == "Correction":
            ax.set_ylabel("Layer", fontsize=25)#, fontsize=20)
        else:
            pass

        # x label appear every 5 ticks

        ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
        ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:], fontsize=25)#, fontsize=20)
        ax.tick_params(axis='x', rotation=0)

        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(0, -len(standardized_scores[0])-2, -5)[::-1][1:], fontsize=25)#, fontsize=20)
        if fig_name == "Correction":
            title_text = "(a) LAT Neural Activity: Correction"
        elif fig_name == "Copy":
            title_text = "(b) LAT Neural Activity: Copy"
        else:
            title_text = f"LAT Neural Activity: {fig_name}"

        # Add title below the plot
        plt.subplots_adjust(top=0.85, bottom=0.05)
        ax.text(0.5, -0.2, title_text, ha='center', va='center', transform=ax.transAxes, fontsize=25)
        
    plt.show()
    return fig
