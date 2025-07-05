#!/bin/bash
# Windows batch script to prepare data for CodeT5+ fine-tuning

echo "Activating virtual environment..."
cd /d "E:\Intel Fest\Fine Tune"
call .\code-t5-env\Scripts\activate

echo "Preparing dataset for fine-tuning..."
python scripts/prepare_data.py ^
    --data_path "E:/Intel Fest/Fine Tune/datasets/clean_code" ^
    --output_path "./data/processed" ^
    --max_samples 50000 ^
    --task_type "code_completion"

echo "Data preparation completed!"
pause
