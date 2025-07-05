"""
Simple training starter script for CodeT5+ fine-tuning
This script handles the environment and starts training
"""

import subprocess
import sys
import os

def run_training():
    """Start the CodeT5+ training process"""
    
    # Change to the correct directory
    os.chdir("E:/Intel Fest/Fine Tune")
    
    # Training command
    training_cmd = [
        sys.executable, "scripts/train_codet5.py",
        "--data_path", "./data/processed_code_completion",
        "--output_dir", "./checkpoints/codet5p-finetuned",
        "--num_train_epochs", "2",
        "--per_device_train_batch_size", "2",
        "--per_device_eval_batch_size", "4",
        "--learning_rate", "3e-5",
        "--warmup_steps", "100",
        "--save_steps", "500",
        "--eval_steps", "250",
        "--logging_steps", "50",
        "--fp16"
    ]
    
    print("🚀 Starting CodeT5+ Fine-tuning...")
    print("=" * 60)
    print(f"Command: {' '.join(training_cmd)}")
    print("=" * 60)
    
    try:
        # Run the training
        result = subprocess.run(training_cmd, check=True, capture_output=False)
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_training()
    if not success:
        print("\n💡 Try running the training manually:")
        print("python scripts/train_codet5.py --data_path ./data/processed_code_completion --output_dir ./checkpoints/codet5p-finetuned --num_train_epochs 2 --per_device_train_batch_size 2 --fp16")
