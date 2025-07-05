"""
Inference script for fine-tuned CodeT5+ model
Test the fine-tuned model on code completion/generation tasks
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import os

class CodeT5Inferencer:
    def __init__(self, model_path, device="auto"):
        """
        Initialize the inferencer with a fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully!")
    
    def generate_code(self, prompt, max_length=256, num_beams=5, temperature=0.8, do_sample=True):
        """
        Generate code based on the given prompt
        
        Args:
            prompt: Input prompt for code generation
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Generated code as string
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output if it's included
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        return generated_text
    
    def interactive_session(self):
        """Run an interactive code generation session"""
        print("\n" + "="*60)
        print("CodeT5+ Interactive Code Generation")
        print("Type 'quit' or 'exit' to end the session")
        print("="*60 + "\n")
        
        while True:
            try:
                prompt = input("Enter your code prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print("\nGenerating code...")
                generated_code = self.generate_code(prompt)
                
                print("\n" + "-"*40)
                print("Generated Code:")
                print("-"*40)
                print(generated_code)
                print("-"*40 + "\n")
                
            except KeyboardInterrupt:
                print("\nSession interrupted by user.")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
        
        print("Session ended.")

def test_model_examples(inferencer):
    """Test the model with some example prompts"""
    test_prompts = [
        "Complete this Python function:\ndef calculate_fibonacci(n):",
        "Write a function to reverse a string:",
        "Complete this code:\nfor i in range(10):",
        "Create a class for a simple calculator:",
        "Write a function to check if a number is prime:"
    ]
    
    print("\n" + "="*60)
    print("Testing Model with Example Prompts")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nExample {i}:")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        try:
            generated_code = inferencer.generate_code(prompt)
            print(f"Generated Code:\n{generated_code}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned CodeT5+ model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on (cpu/cuda/auto)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive session")
    parser.add_argument("--test_examples", action="store_true",
                       help="Test with example prompts")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt for code generation")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        return
    
    # Initialize inferencer
    try:
        inferencer = CodeT5Inferencer(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run based on arguments
    if args.prompt:
        # Single prompt generation
        print(f"Prompt: {args.prompt}")
        result = inferencer.generate_code(args.prompt)
        print(f"Generated Code:\n{result}")
    
    elif args.test_examples:
        # Test with examples
        test_model_examples(inferencer)
    
    elif args.interactive:
        # Interactive session
        inferencer.interactive_session()
    
    else:
        # Default: run test examples
        test_model_examples(inferencer)

if __name__ == "__main__":
    main()
