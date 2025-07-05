# CodeT5+ Fine-tuning Project

This project fine-tunes the CodeT5+ 220M model on clean code data for code completion tasks.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate your virtual environment
.\code-t5-env\Scripts\activate

# Verify CUDA is available
python test.py
```

### 2. Data Preparation
```bash
# Run the data preparation script
.\prepare_data.bat

# Or run manually:
python scripts/prepare_data.py --data_path "E:/Intel Fest/Lint agent M L/datasets/clean_code" --output_path "./data/processed" --max_samples 50000 --task_type "code_completion"
```

### 3. Start Training
```bash
# Run the training script
.\train_model.bat

# Or run manually with custom parameters:
python scripts/train_codet5.py --data_path "./data/processed_code_completion" --output_dir "./checkpoints/codet5p-finetuned" --num_train_epochs 3
```

### 4. Test the Model
```bash
# Interactive testing
.\test_model.bat

# Or test with examples:
python scripts/inference.py --model_path "./checkpoints/codet5p-finetuned" --test_examples
```

## 📁 Project Structure
```
Fine Tune/
├── scripts/
│   ├── prepare_data.py      # Data preprocessing script
│   ├── train_codet5.py      # Main training script
│   └── inference.py         # Model testing and inference
├── data/
│   └── processed/           # Processed training data
├── checkpoints/
│   └── codet5p-finetuned/   # Fine-tuned model output
├── outputs/                 # Training logs and outputs
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── *.bat                   # Batch scripts for Windows
```

## ⚙️ Configuration

Edit `config.py` to modify training parameters:

- **DATA_SOURCE**: Path to your clean_code dataset
- **BATCH_SIZE_TRAIN**: Adjust based on GPU memory (4 for RTX 3050)
- **NUM_EPOCHS**: Number of training epochs (start with 3)
- **MAX_SAMPLES**: Limit samples for testing (set to None for full dataset)

## 🔧 Training Parameters

### For RTX 3050 (4GB VRAM):
- Batch size: 4 (training), 8 (evaluation)
- Mixed precision: Enabled (fp16)
- Gradient accumulation: May be needed for larger effective batch sizes

### Memory Optimization:
- Use smaller batch sizes if you encounter OOM errors
- Enable gradient checkpointing in training args
- Use fp16 mixed precision training

## 📊 Monitoring Training

Training logs and metrics are saved to:
- **TensorBoard logs**: `./checkpoints/codet5p-finetuned/logs/`
- **Training metrics**: Console output and log files

View TensorBoard:
```bash
tensorboard --logdir ./checkpoints/codet5p-finetuned/logs/
```

## 🎯 Supported Tasks

1. **Code Completion**: Complete partial code snippets
2. **Code Summarization**: Generate summaries for code blocks

## 🚨 Troubleshooting

### CUDA Out of Memory:
- Reduce batch size to 2 or 1
- Enable gradient accumulation
- Use smaller max sequence lengths

### Data Loading Issues:
- Check data path in config.py
- Ensure dataset files are accessible
- Verify parquet files are not corrupted

### Model Loading Issues:
- Check internet connection for downloading base model
- Verify sufficient disk space for model caching
- Clear HuggingFace cache if needed: `~/.cache/huggingface/`

## 📈 Expected Results

With proper fine-tuning, you should see:
- Improved ROUGE-L scores on validation set
- Better code completion accuracy
- Coherent and contextually relevant code generations

## 🔄 Next Steps

1. **Start Small**: Use `--max_samples 10000` for initial testing
2. **Monitor Performance**: Check validation loss and ROUGE scores
3. **Adjust Parameters**: Tune learning rate and batch size based on results
4. **Scale Up**: Use full dataset for final training
5. **Evaluate**: Test on your specific use cases

## 📝 Example Usage

```python
from scripts.inference import CodeT5Inferencer

# Load your fine-tuned model
inferencer = CodeT5Inferencer("./checkpoints/codet5p-finetuned")

# Generate code
prompt = "def calculate_fibonacci(n):"
result = inferencer.generate_code(prompt)
print(result)
```

Happy fine-tuning! 🎉
