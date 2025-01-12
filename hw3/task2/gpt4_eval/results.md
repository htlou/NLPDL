# Experiment Setup
For detailed prompt, please refer to `config/system_prompt.py`

# Experiment Results
## Baseline
Accuracy: 1091/1300

## Few-shot(1 shot)
Accuracy: 1185/1300

## Few-shot(2 shot)
Accuracy: 1198/1300

## CoT
Accuracy: 1092/1300

## CoT + Few-shot(1 shot)
Accuracy: 1166/1300

## Reflexion
Accuracy for 3 iterations: 961/1300

This is because the regrex matching is not that working for complicated latex answers.