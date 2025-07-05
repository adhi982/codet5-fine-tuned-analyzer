#!/bin/bash
# Windows batch script to start CodeT5+ fine-tuning

echo "Activating virtual environment..."
cd /d "E:\Intel Fest\Fine Tune"
call .\code-t5-env\Scripts\activate

echo "Starting CodeT5+ fine-tuning..."
python scripts/train_codet5.py ^
    --model_name_or_path "Salesforce/codet5p-220m" ^
    --data_path "./data/processed_code_completion" ^
    --output_dir "./checkpoints/codet5p-finetuned" ^
    --num_train_epochs 3 ^
    --per_device_train_batch_size 4 ^
    --per_device_eval_batch_size 8 ^
    --learning_rate 5e-5 ^
    --weight_decay 0.01 ^
    --warmup_steps 500 ^
    --logging_steps 100 ^
    --save_steps 1000 ^
    --eval_steps 500 ^
    --fp16 ^
    --seed 42

echo "Training completed!"
pause
