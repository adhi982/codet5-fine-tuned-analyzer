"""
Model Verification Script
Tests if the fine-tuned CodeT5+ model is working correctly
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

def check_model_files(model_path):
    """Check if all required model files exist"""
    print(f"\n🔍 Checking model files in: {model_path}")
    print("=" * 60)
    
    required_files = [
        "config.json",           # Model configuration
        "pytorch_model.bin",     # Model weights
        "tokenizer_config.json", # Tokenizer configuration
        "vocab.json",           # Vocabulary
        "merges.txt",           # BPE merges
        "special_tokens_map.json", # Special tokens
        "trainer_state.json",    # Training state
        "training_args.bin"      # Training arguments
    ]
    
    missing_files = []
    existing_files = []
    
    if not os.path.exists(model_path):
        print(f"❌ Model directory does not exist: {model_path}")
        return False
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} ({file_size:,} bytes)")
            existing_files.append(file)
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    # Check for checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
    if checkpoint_dirs:
        print(f"\n📂 Found {len(checkpoint_dirs)} checkpoint directories:")
        for checkpoint in sorted(checkpoint_dirs):
            print(f"   📁 {checkpoint}")
    
    return len(missing_files) == 0

def test_model_loading(model_path):
    """Test if the model can be loaded successfully"""
    print(f"\n🚀 Testing model loading...")
    print("=" * 60)
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size:,}")
        
        # Load model
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {model.num_parameters():,}")
        
        # Check if model is on correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"   Device: {device}")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def test_model_inference(tokenizer, model):
    """Test model inference with sample prompts"""
    print(f"\n🧪 Testing model inference...")
    print("=" * 60)
    
    if tokenizer is None or model is None:
        print("❌ Cannot test inference - model loading failed")
        return False
    
    test_prompts = [
        "Complete this code:\ndef fibonacci(n):",
        "Complete this code:\nfor i in range(10):",
        "Complete this code:\nclass Calculator:",
        "Complete this code:\nimport pandas as pd\ndf = pd.read_csv('data.csv')",
        "Complete this code:\nif __name__ == '__main__':"
    ]
    
    device = next(model.parameters()).device
    model.eval()
    
    print("Testing with sample prompts:")
    success_count = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=3,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input from output if it's included
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            print(f"Generated: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
            print("✅ Generation successful")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print(f"\n📊 Inference Results: {success_count}/{len(test_prompts)} tests passed")
    return success_count == len(test_prompts)

def check_training_logs(model_path):
    """Check training logs and metrics"""
    print(f"\n📊 Checking training logs...")
    print("=" * 60)
    
    # Check trainer state
    trainer_state_path = os.path.join(model_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            
            print("✅ Training completed successfully!")
            print(f"   Total epochs: {trainer_state.get('epoch', 'Unknown')}")
            print(f"   Global step: {trainer_state.get('global_step', 'Unknown')}")
            print(f"   Best metric: {trainer_state.get('best_metric', 'Unknown')}")
            
            # Show training history
            if 'log_history' in trainer_state:
                logs = trainer_state['log_history']
                print(f"   Training logs: {len(logs)} entries")
                
                # Find final metrics
                final_logs = [log for log in logs if 'eval_loss' in log]
                if final_logs:
                    final_log = final_logs[-1]
                    print(f"   Final eval loss: {final_log.get('eval_loss', 'N/A'):.4f}")
                    print(f"   Final ROUGE-L: {final_log.get('eval_rougeL', 'N/A'):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to read training logs: {e}")
            return False
    else:
        print("❌ No training state found")
        return False

def verify_fine_tuned_model(model_path="./checkpoints/codet5p-finetuned"):
    """Main verification function"""
    print("🔍 FINE-TUNED MODEL VERIFICATION")
    print("=" * 60)
    print(f"Model path: {os.path.abspath(model_path)}")
    
    # Test 1: Check files
    files_ok = check_model_files(model_path)
    
    # Test 2: Load model
    tokenizer, model = test_model_loading(model_path)
    
    # Test 3: Test inference
    inference_ok = test_model_inference(tokenizer, model)
    
    # Test 4: Check training logs
    logs_ok = check_training_logs(model_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Model Files", files_ok),
        ("Model Loading", tokenizer is not None and model is not None),
        ("Inference", inference_ok),
        ("Training Logs", logs_ok)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 MODEL VERIFICATION SUCCESSFUL!")
        print("Your fine-tuned CodeT5+ model is working correctly!")
        print("\n💡 You can now use it for:")
        print("   • Code completion")
        print("   • Code generation")
        print("   • Programming assistance")
        print(f"\n📂 Model location: {os.path.abspath(model_path)}")
    else:
        print("⚠️  Some verification tests failed.")
        print("Please check the training process and try again.")
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify fine-tuned CodeT5+ model")
    parser.add_argument("--model_path", type=str, default="./checkpoints/codet5p-finetuned",
                       help="Path to the fine-tuned model")
    
    args = parser.parse_args()
    verify_fine_tuned_model(args.model_path)
