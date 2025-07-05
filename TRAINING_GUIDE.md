# 🎯 CODET5+ FINE-TUNING - FINAL TRAINING COMMANDS

## 📋 Step-by-Step Training Instructions

### 1. Open PowerShell in the project directory
cd "E:\Intel Fest\Fine Tune"

### 2. Activate virtual environment
.\code-t5-env\Scripts\activate

### 3. Verify environment (should show all ✅)
python verify_setup.py

### 4. Start training (choose one option below)

## OPTION A: Quick Training (2 epochs, smaller batch)
python scripts/train_codet5.py --data_path "./data/processed_code_completion" --output_dir "./checkpoints/codet5p-finetuned" --num_train_epochs 2 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --learning_rate 3e-5 --warmup_steps 100 --save_steps 500 --eval_steps 250 --logging_steps 50 --fp16

## OPTION B: Balanced Training (3 epochs, standard batch) 
python scripts/train_codet5.py --data_path "./data/processed_code_completion" --output_dir "./checkpoints/codet5p-finetuned" --num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --learning_rate 5e-5 --warmup_steps 200 --save_steps 250 --eval_steps 125 --logging_steps 25 --fp16

## OPTION C: Full Training (5 epochs, larger dataset)
# First, prepare more data:
python scripts/prepare_data.py --data_path "E:/Intel Fest/Fine Tune/datasets/clean_code" --output_path "./data/processed_large" --max_samples 50000 --task_type "code_completion"

# Then train:
python scripts/train_codet5.py --data_path "./data/processed_large_code_completion" --output_dir "./checkpoints/codet5p-finetuned-large" --num_train_epochs 5 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --learning_rate 5e-5 --warmup_steps 500 --save_steps 1000 --eval_steps 500 --logging_steps 100 --fp16

### 5. Monitor training
# Training logs will appear in console
# TensorBoard logs: ./checkpoints/codet5p-finetuned/logs/
# To view TensorBoard: tensorboard --logdir ./checkpoints/codet5p-finetuned/logs/

### 6. Test the fine-tuned model
python scripts/inference.py --model_path "./checkpoints/codet5p-finetuned" --interactive

## 📊 Expected Training Time (RTX 3050):
- Option A: ~30-45 minutes
- Option B: ~60-90 minutes  
- Option C: ~4-6 hours

## 🔧 If you get memory errors:
- Reduce batch size to 1
- Add --gradient_accumulation_steps 4
- Remove --fp16 flag

## ✅ Training Success Indicators:
- Loss decreasing over time
- ROUGE scores improving
- No CUDA out of memory errors
- Model checkpoints saved regularly

## 🎯 What happens during training:
1. Model loads CodeT5+ 220M base model
2. Preprocesses your code completion data
3. Fine-tunes on code completion task
4. Saves checkpoints every 500 steps
5. Evaluates on validation set
6. Saves final model to ./checkpoints/codet5p-finetuned/

## 🚀 After training completes:
Your fine-tuned model will be ready for:
- Code completion
- Code generation
- Programming assistance
- Integration into your applications
