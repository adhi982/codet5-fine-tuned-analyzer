"""
Simple Code Completion Service
A lightweight service to use your fine-tuned CodeT5+ model
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SimpleCodeComplete:
    """Lightweight wrapper for your fine-tuned model"""
    
    def __init__(self, model_path="E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"):
        print("🚀 Loading your fine-tuned CodeT5+ model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model loaded on {self.device}")
    
    def complete(self, code_snippet, max_length=200):
        """Complete the given code snippet"""
        prompt = f"Complete this code:\n{code_snippet}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the output
        if "Complete this code:" in result:
            result = result.split("Complete this code:")[-1].strip()
        
        return result

# Quick test function
def test_model():
    """Test your fine-tuned model quickly"""
    print("🧪 Testing your fine-tuned model...")
    
    # Initialize the model
    code_completer = SimpleCodeComplete()
    
    # Test cases
    test_cases = [
        "def fibonacci(n):",
        "for i in range(10):",
        "class DataProcessor:",
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
        "if __name__ == '__main__':"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {test_case}")
        print("Output:")
        try:
            result = code_completer.complete(test_case)
            print(result)
            print("✅ Success")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Model testing completed!")

if __name__ == "__main__":
    test_model()
