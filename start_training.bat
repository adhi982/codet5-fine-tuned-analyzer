@echo off
cd /d "E:\Intel Fest\Fine Tune"
call .\code-t5-env\Scripts\activate.bat

echo Starting CodeT5+ fine-tuning...
python scripts/train_codet5.py ^
    --data_path "./data/processed_code_completion" ^
    --output_dir "./checkpoints/codet5p-finetuned" ^
    --num_train_epochs 2 ^
    --per_device_train_batch_size 2 ^
    --per_device_eval_batch_size 4 ^
    --learning_rate 3e-5 ^
    --warmup_steps 100 ^
    --save_steps 500 ^
    --eval_steps 250 ^
    --logging_steps 50 ^
    --fp16

echo Training completed!
pause
