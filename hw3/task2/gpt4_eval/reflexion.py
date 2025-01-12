import json
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse
from utils_openai import request_openai_noexcept, get_openai_api_keys
import ray
from tqdm import tqdm

REFLEXION_SYSTEM_PROMPT = """You are a mathematical reasoning expert. Your task is to:
1. Review the original math question
2. Review your previous solution attempt
3. Check if your previous solution was correct
4. If you find any errors, provide a new solution
5. If your previous solution was correct, explain why

Write your final answer after the #### sign at the end of your response."""

REFLEXION_USER_PROMPT = """Original Question:
{question}

Your Previous Solution:
{previous_solution}

Please reflect on your solution and provide a new answer if needed:"""

def extract_answer(response: str) -> str:
    """Extract the final answer after the #### sign."""
    pattern = r'####(.*)'
    matches = re.findall(pattern, response, re.IGNORECASE)
    return matches[-1] if matches else None

@ray.remote
def process_single_question(
    question: Dict[str, str],
    openai_api_key: str,
    openai_model: str,
    base_url: str,
    max_iterations: int
) -> Dict[str, Any]:
    """Process a single question with multiple reflection iterations."""
    
    # Initial solution
    messages = [
        {'role': 'system', 'content': "You are a math expert. Write your answer after the #### sign."},
        {'role': 'user', 'content': f"Solve this math problem: {question['question']}"}
    ]
    
    current_solution = request_openai_noexcept(
        messages=messages,
        openai_api_keys=openai_api_key,
        openai_model=openai_model,
        base_url=base_url
    )
    
    solutions = [current_solution['output']]
    
    # Reflection iterations
    for i in range(max_iterations - 1):
        messages = [
            {'role': 'system', 'content': REFLEXION_SYSTEM_PROMPT},
            {'role': 'user', 'content': REFLEXION_USER_PROMPT.format(
                question=question['question'],
                previous_solution=current_solution['output']
            )}
        ]
        
        current_solution = request_openai_noexcept(
            messages=messages,
            openai_api_keys=openai_api_key,
            openai_model=openai_model,
            base_url=base_url
        )
        
        solutions.append(current_solution['output'])
    
    # Extract final answers from all iterations
    answers = [extract_answer(sol) for sol in solutions]
    
    return {
        'question': question['question'],
        'ground_truth': question['answer'],
        'solutions': solutions,
        'answers': answers,
        'final_answer': answers[-1]
    }

def run_reflexion(
    input_file: str,
    output_dir: str,
    max_iterations: int = 3,
    openai_api_key_file: str = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    num_workers: int = 4
) -> None:
    """Run reflexion process on GSM8K dataset."""
    
    # Initialize Ray
    ray.init()
    
    # Load questions
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    # Get API keys
    api_keys = get_openai_api_keys([], openai_api_key_file)
    api_key = api_keys[0][0]  # Use first API key
    
    # Process questions in parallel
    futures = [
        process_single_question.remote(
            question,
            api_key,
            model,
            base_url,
            max_iterations
        )
        for question in questions
    ]
    
    # Collect results with progress bar
    results = []
    for future in tqdm(futures, total=len(futures)):
        result = ray.get(future)
        results.append(result)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"gsm8k_reflexion_{max_iterations}iterations.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate accuracy for each iteration
    accuracies = []
    for i in range(max_iterations):
        correct = sum(1 for r in results if extract_answer(r['ground_truth']) == r['answers'][i])
        accuracy = correct / len(results)
        accuracies.append(accuracy)
        print(f"Iteration {i+1} Accuracy: {accuracy:.4f}")
    
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--openai-api-key-file", type=str, default="config/openai_api_keys.txt")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    
    run_reflexion(
        input_file=args.input_file,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        openai_api_key_file=args.openai_api_key_file,
        base_url=args.base_url,
        model=args.model,
        num_workers=args.num_workers
    )
