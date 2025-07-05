"""
CodeT5+ Fine-tuned Model Integration Examples
Shows how to use your fine-tuned model in different projects
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class CodeT5Assistant:
    """
    A simple wrapper class for your fine-tuned CodeT5+ model
    Use this in your projects for code completion and generation
    """
    
    def __init__(self, model_path="E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"):
        """
        Initialize the code assistant with your fine-tuned model
        
        Args:
            model_path: Path to your fine-tuned model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CodeT5+ model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
    
    def complete_code(self, prompt, max_length=256, num_beams=3, temperature=0.8, do_sample=True):
        """
        Complete code based on the given prompt
        
        Args:
            prompt: Incomplete code to complete
            max_length: Maximum length of completion
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Completed code as string
        """
        # Prepare input
        if not prompt.startswith("Complete this code:"):
            prompt = f"Complete this code:\n{prompt}"
        
        # Tokenize
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
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up output
        if "Complete this code:" in generated_text:
            generated_text = generated_text.split("Complete this code:")[-1].strip()
        
        return generated_text
    
    def generate_function(self, function_name, description=""):
        """
        Generate a complete function based on name and description
        
        Args:
            function_name: Name of the function to generate
            description: Optional description of what the function should do
        
        Returns:
            Generated function code
        """
        if description:
            prompt = f"Complete this code:\ndef {function_name}():  # {description}"
        else:
            prompt = f"Complete this code:\ndef {function_name}():"
        
        return self.complete_code(prompt)
    
    def complete_class(self, class_name, base_class=""):
        """
        Generate a class structure
        
        Args:
            class_name: Name of the class
            base_class: Optional base class
        
        Returns:
            Generated class code
        """
        if base_class:
            prompt = f"Complete this code:\nclass {class_name}({base_class}):"
        else:
            prompt = f"Complete this code:\nclass {class_name}:"
        
        return self.complete_code(prompt)
    
    def fix_syntax(self, broken_code):
        """
        Attempt to fix syntax errors in code
        
        Args:
            broken_code: Code with potential syntax errors
        
        Returns:
            Fixed code
        """
        prompt = f"Fix this code:\n{broken_code}"
        return self.complete_code(prompt)


# Example usage functions
def example_basic_completion():
    """Example: Basic code completion"""
    print("🔧 Example 1: Basic Code Completion")
    print("=" * 50)
    
    assistant = CodeT5Assistant()
    
    incomplete_code = """def calculate_fibonacci(n):
    if n <= 1:
        return n"""
    
    completed = assistant.complete_code(incomplete_code)
    print(f"Input:\n{incomplete_code}")
    print(f"\nCompleted:\n{completed}")
    print()

def example_function_generation():
    """Example: Generate complete functions"""
    print("🔧 Example 2: Function Generation")
    print("=" * 50)
    
    assistant = CodeT5Assistant()
    
    # Generate a sorting function
    function_code = assistant.generate_function("bubble_sort", "sort a list using bubble sort algorithm")
    print("Generated function:")
    print(function_code)
    print()

def example_class_generation():
    """Example: Generate class structures"""
    print("🔧 Example 3: Class Generation")
    print("=" * 50)
    
    assistant = CodeT5Assistant()
    
    # Generate a calculator class
    class_code = assistant.complete_class("Calculator")
    print("Generated class:")
    print(class_code)
    print()

def example_api_integration():
    """Example: Simple API for code completion"""
    print("🔧 Example 4: API Integration")
    print("=" * 50)
    
    from flask import Flask, request, jsonify
    
    # Initialize the assistant globally
    assistant = CodeT5Assistant()
    app = Flask(__name__)
    
    @app.route('/complete', methods=['POST'])
    def complete_code_api():
        """API endpoint for code completion"""
        data = request.json
        code_prompt = data.get('code', '')
        
        if not code_prompt:
            return jsonify({'error': 'No code provided'}), 400
        
        try:
            completed_code = assistant.complete_code(code_prompt)
            return jsonify({
                'input': code_prompt,
                'completed': completed_code,
                'status': 'success'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/generate_function', methods=['POST'])
    def generate_function_api():
        """API endpoint for function generation"""
        data = request.json
        function_name = data.get('name', '')
        description = data.get('description', '')
        
        if not function_name:
            return jsonify({'error': 'No function name provided'}), 400
        
        try:
            function_code = assistant.generate_function(function_name, description)
            return jsonify({
                'function_name': function_name,
                'description': description,
                'code': function_code,
                'status': 'success'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    print("Flask API example created!")
    print("Endpoints:")
    print("  POST /complete - Complete code")
    print("  POST /generate_function - Generate functions")
    print()

if __name__ == "__main__":
    print("🚀 CodeT5+ Fine-tuned Model Integration Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_basic_completion()
        example_function_generation()
        example_class_generation()
        example_api_integration()
        
        print("✅ All examples completed successfully!")
        print("\n💡 You can now use the CodeT5Assistant class in your projects!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure your fine-tuned model is available at the specified path.")
