"""
CodeT5+ Fine-tuning Script
Fine-tunes CodeT5+ 220M model on clean code dataset
"""

import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path

from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed
)
import numpy as np
from rouge_score import rouge_scorer
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to model/config/tokenizer."""
    model_name_or_path: str = field(
        default="Salesforce/codet5p-220m",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to data."""
    data_path: str = field(
        metadata={"help": "Path to the training data"}
    )
    max_source_length: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length"}
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for preprocessing"}
    )

class CodeT5Trainer:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Set random seed
        set_seed(training_args.seed)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True
        )
        
        # Setup special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Model loaded: {model_args.model_name_or_path}")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        logger.info(f"Loading data from {self.data_args.data_path}")
        
        # Load dataset
        if os.path.isdir(self.data_args.data_path):
            dataset = load_from_disk(self.data_args.data_path)
        else:
            raise ValueError(f"Data path {self.data_args.data_path} not found")
        
        # Preprocessing function
        def preprocess_function(examples):
            inputs = examples['input_text']
            targets = examples['target_text']
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.data_args.max_source_length,
                padding=False,
                truncation=True
            )
            
            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.data_args.max_target_length,
                    padding=False,
                    truncation=True
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        
        train_dataset = dataset['train'].map(
            preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=dataset['train'].column_names,
            desc="Preprocessing train dataset"
        )
        
        eval_dataset = None
        if 'validation' in dataset:
            eval_dataset = dataset['validation'].map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=dataset['validation'].column_names,
                desc="Preprocessing validation dataset"
            )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        try:
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            for pred, label in zip(decoded_preds, decoded_labels):
                scores = scorer.score(label, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            result = {
                'rouge1': np.mean(rouge_scores['rouge1']),
                'rouge2': np.mean(rouge_scores['rouge2']),
                'rougeL': np.mean(rouge_scores['rougeL'])
            }
        except:
            result = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        
        return result
    
    def train(self):
        """Main training function"""
        # Load and preprocess data
        train_dataset, eval_dataset = self.load_and_preprocess_data()
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        trainer.save_state()
        
        # Training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluation
        if eval_dataset:
            logger.info("Running evaluation...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        
        logger.info("Training completed!")
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5+ on clean code dataset")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/codet5p-220m",
                       help="Model name or path")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for models")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to preprocessed data")
    parser.add_argument("--max_source_length", type=int, default=512,
                       help="Maximum source length")
    parser.add_argument("--max_target_length", type=int, default=256,
                       help="Maximum target length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/codet5p-finetuned",
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Create argument objects
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    
    data_args = DataArguments(
        data_path=args.data_path,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        seed=args.seed,
        fp16=args.fp16,
        report_to=["tensorboard"],
        logging_dir=f"{args.output_dir}/logs",
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        save_total_limit=3,
        remove_unused_columns=False
    )
    
    # Initialize trainer and start training
    trainer_obj = CodeT5Trainer(model_args, data_args, training_args)
    trainer_obj.train()

if __name__ == "__main__":
    main()
