from dataclasses import dataclass, field
from typing import Optional
import logging
import random
import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate
import wandb
from dataHelper import get_dataset
from transformers.trainer_callback import TrainerCallback

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval
    """
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use"}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={"help": "Max sequence length for tokenization"}
    )
    num_shots: Optional[int] = field(
        default=None,
        metadata={"help": "Number of shots for few-shot learning"}
    )

class WandbCallback(TrainerCallback):
    """Custom callback for logging training metrics to wandb."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

def main():
    '''
    Initialize logging, seed, and parse arguments
    '''
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    '''
    Load and process dataset
    '''
    # Load dataset using helper function
    dataset, label_map = get_dataset(data_args.dataset_name, sep_token="<sep>", 
                                   num_shots=data_args.num_shots, verbose=True)
    num_labels = len(set(dataset['train']['label']))

    '''
    Load model and tokenizer
    '''
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    '''
    Process datasets and build data collator
    '''
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "label"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    '''
    Setup metrics
    '''

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy_metric = evaluate.load("accuracy")
        f1_macro_metric = evaluate.load("f1")
        f1_micro_metric = evaluate.load("f1")

        # Calculate metrics
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
        micro_f1 = f1_micro_metric.compute(predictions=predictions, references=labels, average='micro')['f1']
        macro_f1 = f1_macro_metric.compute(predictions=predictions, references=labels, average='macro')['f1']
        
        return {
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }

    '''
    Initialize wandb and trainer
    '''
    wandb.init(project="nlp-classification", name=f"{model_args.model_name_or_path}-{data_args.dataset_name}-{training_args.output_dir}")

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[WandbCallback()],
    )

    '''
    Training and evaluation
    '''
    train_result = trainer.train()
    
    metrics = trainer.evaluate()
    wandb.log(metrics)

    # Log final metrics
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()