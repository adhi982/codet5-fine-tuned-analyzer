"""
Ready-to-Use CodeT5+ Assistant
Copy this file to any of your projects to use your fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class CodeT5Assistant:
    """Your fine-tuned CodeT5+ model ready for any project"""
    
    def __init__(self, model_path="E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"):
        """Initialize with your fine-tuned model"""
        print("🚀 Loading your fine-tuned CodeT5+ model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def complete_code(self, code_prompt, max_length=200, clean_output=True):
        """
        Complete code based on the given prompt
        
        Args:
            code_prompt: The incomplete code to complete
            max_length: Maximum length of the completion
            clean_output: Whether to clean the output
        
        Returns:
            Completed code as string
        """
        # Format the prompt
        if not code_prompt.startswith("Complete this code:"):
            formatted_prompt = f"Complete this code:\n{code_prompt}"
        else:
            formatted_prompt = code_prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
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
                num_beams=3,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean output if requested
        if clean_output:
            if "Complete this code:" in result:
                result = result.split("Complete this code:")[-1].strip()
        
        return result
    
    def generate_function(self, function_name, description="", parameters=""):
        """Generate a complete function"""
        if parameters:
            prompt = f"def {function_name}({parameters}):"
        else:
            prompt = f"def {function_name}():"
        
        if description:
            prompt += f"  # {description}"
        
        return self.complete_code(prompt)
    
    def complete_class(self, class_name, base_class=""):
        """Generate a class structure"""
        if base_class:
            prompt = f"class {class_name}({base_class}):"
        else:
            prompt = f"class {class_name}:"
        
        return self.complete_code(prompt)


# Example usage functions
def demo_basic_usage():
    """Demo: Basic usage examples"""
    print("\n🔧 DEMO: Basic Usage")
    print("=" * 50)
    
    # Initialize assistant
    assistant = CodeT5Assistant()
    
    # Example 1: Function completion
    print("Example 1: Complete a function")
    incomplete_function = "def fibonacci(n):\n    if n <= 1:"
    completed = assistant.complete_code(incomplete_function)
    print(f"Input:\n{incomplete_function}")
    print(f"Completed:\n{completed}")
    
    # Example 2: Generate new function
    print("\nExample 2: Generate a function")
    new_function = assistant.generate_function("binary_search", "search for element in sorted array", "arr, target")
    print(f"Generated function:\n{new_function}")
    
    # Example 3: Class generation
    print("\nExample 3: Generate a class")
    new_class = assistant.complete_class("Calculator")
    print(f"Generated class:\n{new_class}")

def demo_project_integration():
    """Demo: How to use in your projects"""
    print("\n🚀 DEMO: Project Integration")
    print("=" * 50)
    
    assistant = CodeT5Assistant()
    
    # Simulate IDE code completion
    current_code = "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf."
    completion = assistant.complete_code(current_code)
    print(f"IDE Completion Example:\n{current_code}")
    print(f"Suggested completion:\n{completion}")

# Quick test function
def quick_test():
    """Quick test to verify everything works"""
    print("🧪 Quick Test of Your Fine-tuned Model")
    print("=" * 50)
    
    try:
        assistant = CodeT5Assistant()
        
        test_code = "def hello_world():"
        result = assistant.complete_code(test_code)
        
        print(f"✅ Test successful!")
        print(f"Input: {test_code}")
        print(f"Output: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Your Fine-tuned CodeT5+ Assistant")
    print("=" * 60)
    
    # Run quick test first
    if quick_test():
        print("\n" + "=" * 60)
        print("📚 Ready to use in your projects!")
        print("\n💡 Copy this file to any project and use:")
        print("   from ready_to_use_assistant import CodeT5Assistant")
        print("   assistant = CodeT5Assistant()")
        print("   result = assistant.complete_code('def my_function():')")
        
        # Run demos
        demo_basic_usage()
        demo_project_integration()
    else:
        print("Please check your model path and try again.")
