"""
ULTRA-SIMPLE: Just run this file to use your fine-tuned model!
No setup needed - just run and start coding!
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    """Load your fine-tuned CodeT5+ model"""
    print("🚀 Loading your fine-tuned CodeT5+ model...")
    
    model_path = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return tokenizer, model, device

def complete_code(tokenizer, model, device, code):
    """Complete the given code"""
    prompt = f"Complete this code:\\n{code}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Complete this code:")[-1].strip()

def main():
    # Load model once
    tokenizer, model, device = load_model()
    
    print("\\n" + "="*60)
    print("🎯 YOUR FINE-TUNED MODEL IS READY!")
    print("="*60)
    print("Type 'quit' to exit, or enter code to complete")
    print("Examples: 'def hello():', 'class Person:', 'for i in range(10):'")
    print("="*60)
    
    while True:
        try:
            # Get user input
            user_code = input("\\n💻 Enter code to complete: ").strip()
            
            if user_code.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_code:
                continue
            
            # Complete the code
            print("🤖 Generating completion...")
            completion = complete_code(tokenizer, model, device, user_code)
            
            print("✨ Completion:")
            print("-" * 40)
            print(completion)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
