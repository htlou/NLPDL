import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from customized_gpt2 import CustomizedGPT2LMHeadModel

device_0 = 'cuda:0'
device_1 = 'cuda:1'

torch.set_printoptions(threshold=10000)

@torch.no_grad()
def customized_greedy_decoding(batch):
# Tokenize input batch
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device_1)
    input_ids = tokenized_batch['input_ids']
    res = input_ids
    attention_mask = tokenized_batch['attention_mask']

    past_key_values = None  # Initialize kv cache
    start_time = time.time()
    get_last_token = False
        
    for timestep in range(MAX_NEW_LENGTH):
        # Pass past_key_values for incremental decoding

        outputs = custom_model(
            input_ids = res,
            # input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,  # Enable caching
            get_last_token = get_last_token
        )
        
        get_last_token = True
        # Extract logits and predicted token
        logits = outputs['logits']
        past_key_values = outputs['past_key_values']  # Update kv cache
        output_tokens = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        res = torch.cat([res, output_tokens], dim=-1)


        # Prepare input for the next step
        input_ids = output_tokens  # Only pass the last predicted token
        attention_mask = torch.cat([attention_mask, torch.ones_like(output_tokens)], dim=-1)

    return res, time.time() - start_time

@torch.no_grad()
def customized_greedy_decoding_wo_cache(batch):
    # Tokenize input batch
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device_1)
    input_ids = tokenized_batch['input_ids']
    attention_mask = tokenized_batch['attention_mask']

    res = input_ids

    past_key_values = None  # Initialize kv cache
    start_time = time.time()
    sequence_len = input_ids.shape[1]
    for timestep in range(sequence_len-1):
        input_ids_timestep = input_ids[:, timestep:timestep+1]
        attention_mask_timestep = attention_mask[:, :timestep+1]
        outputs = custom_model(
            input_ids = input_ids_timestep,
            attention_mask=attention_mask_timestep,
            past_key_values=past_key_values,
            use_cache=True,  # Enable caching
            get_last_token = True
        )
        logits = outputs['logits']
        past_key_values = outputs['past_key_values'] 

        
    for timestep in range(MAX_NEW_LENGTH):

        outputs = custom_model(
            input_ids = res,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,  # Enable caching
            get_last_token = True
        )
        
        get_last_token = True
        logits = outputs['logits']
        past_key_values = outputs['past_key_values']  # Update kv cache
        output_tokens = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        res = torch.cat([res, output_tokens], dim=-1)

        # Prepare input for the next step
        input_ids = output_tokens  # Only pass the last predicted token
        attention_mask = torch.cat([attention_mask, torch.ones_like(output_tokens)], dim=-1)

    return res, time.time() - start_time


@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device_0)
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in tqdm(range(MAX_NEW_LENGTH), desc='golden'):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 16
    times = [0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map=device_0)
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map=device_1)
    custom_model.eval()

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in tqdm(range(0, (len(prompt_dataset) + bsz - 1) // bsz), desc="Processing batches"):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        custom_res, custom_time = customized_greedy_decoding(batch)
        torch.cuda.empty_cache()
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        torch.cuda.empty_cache()

        times[0] += golden_wo_cache_time
        times[1] += custom_time

        # assert torch.equal(golden_wo_cache_res.to('cpu'), custom_res.to('cpu')), "Decoding results are not equal, expected: {}, shape: {}, got: {}, shape: {}".format(golden_wo_cache_res, golden_wo_cache_res.shape, custom_res, custom_res.shape)
        if not torch.equal(golden_wo_cache_res.to('cpu'), custom_res.to('cpu')):
            # for custom_list, golden_list in zip(custom_res, golden_wo_cache_res):
            #     custom_text = tokenizer.decode(custom_list, skip_special_tokens=True)
            #     golden_text = tokenizer.decode(golden_list, skip_special_tokens=True)
            #     print("Decoding results are not equal, expected: {}".format(golden_text))
            #     print("got: {}".format(custom_text))
            diff = torch.abs(golden_wo_cache_res.to('cpu') - custom_res.to('cpu'))
            print("Difference: ", diff)
            exit()

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])