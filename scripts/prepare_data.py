"""
Data preparation script for CodeT5+ fine-tuning
Prepares the clean_code dataset for training
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import json
from pathlib import Path
import argparse

def load_and_process_data(data_path, output_path, max_samples=None, test_size=0.1):
    """
    Load and process the clean code dataset
    
    Args:
        data_path: Path to the dataset (can be local parquet files or HuggingFace dataset)
        output_path: Where to save processed data
        max_samples: Maximum number of samples to use (for testing/development)
        test_size: Fraction of data to use for validation
    """
    print("Loading dataset...")
    
    # Try to load from local path first, then from HuggingFace
    try:
        if os.path.exists(data_path):
            # Load from local dataset files (Arrow format or parquet)
            if os.path.exists(os.path.join(data_path, "dataset_dict.json")):
                # Dataset is already in HuggingFace format
                dataset = load_dataset(data_path)
            else:
                # Try to load as parquet files
                dataset = load_dataset("parquet", data_files=f"{data_path}/**/*.parquet")
        else:
            # Load from HuggingFace
            dataset = load_dataset(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your dataset path is correct")
        return None
    
    print(f"Dataset loaded. Available splits: {list(dataset.keys())}")
    
    # Get the training data
    if 'train' in dataset:
        train_data = dataset['train']
    else:
        train_data = dataset[list(dataset.keys())[0]]
    
    print(f"Total samples: {len(train_data)}")
    
    # Limit samples if specified (useful for testing)
    if max_samples and max_samples < len(train_data):
        train_data = train_data.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    # Examine the data structure
    print("Dataset columns:", train_data.column_names)
    print("Sample data:")
    for i in range(min(3, len(train_data))):
        sample = train_data[i]
        print(f"Sample {i}:")
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Create train/validation split
    split_dataset = train_data.train_test_split(test_size=test_size, seed=42)
    
    # Save processed dataset
    os.makedirs(output_path, exist_ok=True)
    split_dataset.save_to_disk(output_path)
    
    print(f"Dataset saved to {output_path}")
    print(f"Train samples: {len(split_dataset['train'])}")
    print(f"Validation samples: {len(split_dataset['test'])}")
    
    return split_dataset

def prepare_code_tasks(dataset, task_type="code_completion"):
    """
    Prepare the dataset for specific code tasks
    
    Args:
        dataset: The loaded dataset
        task_type: Type of task (code_completion, bug_fixing, code_summarization)
    """
    def process_sample(example):
        if task_type == "code_completion":
            # For code completion: input is partial code, output is complete code
            if 'code' in example:
                code = example['code']
                if len(code.split('\n')) > 5:  # Only use multi-line code
                    lines = code.split('\n')
                    split_point = len(lines) // 2
                    input_code = '\n'.join(lines[:split_point])
                    target_code = '\n'.join(lines[split_point:])
                    
                    return {
                        'input_text': f"Complete this code:\n{input_code}",
                        'target_text': target_code
                    }
        
        elif task_type == "code_summarization":
            # For code summarization: input is code, output is summary/docstring
            if 'code' in example:
                return {
                    'input_text': f"Summarize this code:\n{example['code']}",
                    'target_text': f"This code performs a specific function"
                }
        
        return None
    
    # Filter and process samples
    processed_samples = []
    print(f"Processing {len(dataset)} samples for {task_type}...")
    
    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} samples...")
        
        processed = process_sample(example)
        if processed:
            processed_samples.append(processed)
        
        # Limit the number of processed samples to avoid memory issues
        if len(processed_samples) >= 5000:  # Process max 5000 samples for now
            break
    
    print(f"Successfully processed {len(processed_samples)} samples")
    
    if len(processed_samples) == 0:
        print("Warning: No samples were processed. Check your data format.")
        return None
    
    return Dataset.from_pandas(pd.DataFrame(processed_samples))

def main():
    parser = argparse.ArgumentParser(description="Prepare data for CodeT5+ fine-tuning")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default="./data/processed", help="Output path for processed data")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--task_type", type=str, default="code_completion", 
                       choices=["code_completion", "code_summarization"], 
                       help="Type of task to prepare data for")
    
    args = parser.parse_args()
    
    # Load and process data
    dataset = load_and_process_data(args.data_path, args.output_path, args.max_samples)
    
    if dataset:
        print("Data preparation completed successfully!")
        
        # Prepare for specific tasks
        if args.task_type:
            print(f"Preparing data for {args.task_type} task...")
            train_processed = prepare_code_tasks(dataset['train'], args.task_type)
            val_processed = prepare_code_tasks(dataset['test'], args.task_type)
            
            processed_dataset = DatasetDict({
                'train': train_processed,
                'validation': val_processed
            })
            
            task_output_path = f"{args.output_path}_{args.task_type}"
            processed_dataset.save_to_disk(task_output_path)
            print(f"Task-specific dataset saved to {task_output_path}")

if __name__ == "__main__":
    main()
