import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def measure_everything(model, tokenizer, input_text, max_new_tokens, use_cache, num_repeats):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    baseline_memories = []
    final_memories = []
    peak_memories = []
    times = []
    token_nums = []

    for _ in range(num_repeats):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start_time = time.time()
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        past_key_values = None

        baseline_memory = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids, use_cache=use_cache, past_key_values=past_key_values)
                input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                if use_cache:
                    past_key_values = outputs.past_key_values

        end_time = time.time()
        times.append(end_time - start_time)

        final_memory = torch.cuda.memory_allocated(device)
        peak_memory = torch.cuda.max_memory_allocated(device)

        baseline_memories.append(baseline_memory)
        final_memories.append(final_memory)
        peak_memories.append(peak_memory)
        token_nums.append(outputs.logits.shape[1])
        
    avg_baseline = sum(baseline_memories) / num_repeats
    avg_final = sum(final_memories) / num_repeats
    avg_peak = sum(peak_memories) / num_repeats
    avg_time = sum(times) / num_repeats
    avg_token_num = sum(token_nums) / num_repeats

    return avg_baseline, avg_final, avg_peak, avg_time, avg_token_num

def test_quantization_memory(model_name, input_text, max_new_tokens, num_repeats):
    results = {}
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['baseline'] = measure_everything(model, tokenizer, input_text, max_new_tokens, use_cache=False, num_repeats=num_repeats)

    results['kv_cache'] = measure_everything(model, tokenizer, input_text, max_new_tokens, use_cache=True, num_repeats=num_repeats)

    # fp16
    quant_config_fp16 = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=False,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map,quantization_config=quant_config_fp16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['fp16'] = measure_everything(model, tokenizer, input_text, max_new_tokens, use_cache=True, num_repeats=num_repeats)

    # int8
    quant_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['int8'] = measure_everything(model, tokenizer, input_text, max_new_tokens, use_cache=True, num_repeats=num_repeats)

    return results

if __name__ == "__main__":
    model_name = "openai-community/gpt2"
    input_text = "Hello, this is JF Kennedy in the loo. Say something to me."
    max_new_tokens = 512
    num_repeats = 10

    memory_results = test_quantization_memory(model_name, input_text, max_new_tokens, num_repeats)

    print(f"Memory results:")
    for config, (baseline, final, peak, time, token_num) in memory_results.items():
        print(f"config: {config}")
        print(f"baseline: {baseline / 1024**2:.2f} MB")
        print(f"final: {final / 1024**2:.2f} MB")
        print(f"peak: {peak / 1024**2:.2f} MB")
        print(f"repeat: {num_repeats} times")
        print(f"avg time: {time:.2f} s")
        print(f"avg token num: {token_num}")
        print()