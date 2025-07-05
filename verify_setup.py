"""
Setup verification script
Tests if all dependencies and environment are properly configured
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
        return False
    
    try:
        from rouge_score import rouge_scorer
        print("✅ Rouge Score")
    except ImportError as e:
        print(f"❌ Rouge Score import failed: {e}")
        return False
    
    try:
        import accelerate
        print(f"✅ Accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"❌ Accelerate import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the base CodeT5+ model can be loaded"""
    print("\nTesting CodeT5+ model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        model_name = "Salesforce/codet5p-220m"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model parameters: {model.num_parameters():,}")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size:,}")
        
        # Test a simple inference
        test_input = "def hello_world():"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=2)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Test generation: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_path():
    """Test if data path exists and is accessible"""
    print("\nTesting data path...")
    
    data_path = "E:/Intel Fest/Fine Tune/datasets/clean_code"
    
    if os.path.exists(data_path):
        print(f"✅ Data path exists: {data_path}")
        
        # List some files
        try:
            files = os.listdir(data_path)
            parquet_files = [f for f in files if f.endswith('.parquet')]
            print(f"   Found {len(parquet_files)} parquet files")
            if parquet_files:
                print(f"   Example files: {parquet_files[:3]}")
        except Exception as e:
            print(f"⚠️  Could not list files: {e}")
        
        return True
    else:
        print(f"❌ Data path not found: {data_path}")
        print("   Please check the path or copy your data to the project directory")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting project directories...")
    
    required_dirs = ["scripts", "data", "checkpoints", "outputs"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False
    
    return True

def main():
    print("🔍 CodeT5+ Fine-tuning Setup Verification")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Project Directories", test_directories),
        ("Data Path", test_data_path),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your environment is ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Run: .\\prepare_data.bat")
        print("2. Run: .\\train_model.bat")
        print("3. Run: .\\test_model.bat")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting:")
        print("- Check if you're in the correct virtual environment")
        print("- Verify CUDA installation if GPU tests failed")
        print("- Check data path and file permissions")
        print("- Ensure all required packages are installed")

if __name__ == "__main__":
    main()
