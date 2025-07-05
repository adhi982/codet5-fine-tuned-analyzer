#!/bin/bash
# Windows batch script to test the fine-tuned model

echo "Activating virtual environment..."
cd /d "E:\Intel Fest\Fine Tune"
call .\code-t5-env\Scripts\activate

echo "Starting interactive testing session..."
python scripts/inference.py ^
    --model_path "./checkpoints/codet5p-finetuned" ^
    --interactive

pause
