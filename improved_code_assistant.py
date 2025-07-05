"""
IMPROVED CODE ASSISTANT: Better generation parameters
Copy this to your projects for cleaner code completions
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ImprovedCodeAssistant:
    def __init__(self, model_path="E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"):
        """Load your fine-tuned model with optimized settings"""
        print("🚀 Loading your fine-tuned CodeT5+ model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def complete_code(self, code_prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """
        Complete code with improved generation parameters
        
        Args:
            code_prompt: The incomplete code
            max_new_tokens: Maximum new tokens to generate
            temperature: Controls randomness (0.1-1.0, lower = more focused)
            top_p: Controls diversity (0.1-1.0, lower = more focused)
        """
        # Format prompt
        if not code_prompt.strip().startswith("Complete this code:"):
            formatted_prompt = f"Complete this code:\n{code_prompt}"
        else:
            formatted_prompt = code_prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=400,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate with improved parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_beams=1,  # Faster generation
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                repetition_penalty=1.2,  # Reduce repetition
                no_repeat_ngram_size=3   # Avoid repeating 3-grams
            )
        
        # Decode and clean
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the output
        if "Complete this code:" in result:
            result = result.split("Complete this code:")[-1].strip()
        
        # Remove the original prompt if it appears in output
        lines = result.split('\n')
        if lines and lines[0].strip() == code_prompt.strip():
            result = '\n'.join(lines[1:])
        
        return result.strip()
    
    def generate_multiple_options(self, code_prompt, num_options=3):
        """Generate multiple completion options"""
        options = []
        for i in range(num_options):
            # Vary temperature for different options
            temp = 0.5 + (i * 0.2)  # 0.5, 0.7, 0.9
            completion = self.complete_code(code_prompt, temperature=temp, max_new_tokens=80)
            if completion and completion not in options:
                options.append(completion)
        return options
    
    def complete_function(self, function_signature, max_new_tokens=150):
        """Complete a function definition"""
        return self.complete_code(function_signature, max_new_tokens=max_new_tokens, temperature=0.6)
    
    def complete_class(self, class_definition, max_new_tokens=200):
        """Complete a class definition"""
        return self.complete_code(class_definition, max_new_tokens=max_new_tokens, temperature=0.6)

# ================================
# USAGE EXAMPLES
# ================================

def demo_improved_assistant():
    """Demo with better generation"""
    assistant = ImprovedCodeAssistant()
    
    print("\\n" + "="*60)
    print("🎯 IMPROVED CODE COMPLETION DEMO")
    print("="*60)
    
    # Test cases
    test_cases = [
        "def fibonacci(n):",
        "class Calculator:\\n    def __init__(self):",
        "def merge_sort(arr):",
        "import requests\\n\\ndef fetch_data(url):",
        "for i in range(10):"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n📝 Example {i}")
        print("-" * 40)
        print(f"Input:\\n{test_case}")
        print("\\nOutput:")
        
        try:
            completion = assistant.complete_code(test_case)
            print(completion)
        except Exception as e:
            print(f"Error: {e}")
    
    # Multiple options demo
    print("\\n🔄 Multiple Options Demo")
    print("-" * 40)
    test_prompt = "def calculate_area(length, width):"
    print(f"Input: {test_prompt}")
    print("\\nDifferent completion options:")
    
    options = assistant.generate_multiple_options(test_prompt)
    for i, option in enumerate(options, 1):
        print(f"\\nOption {i}:")
        print(option)

# ================================
# INTEGRATION TEMPLATES
# ================================

def create_web_api():
    """Flask API template"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    assistant = ImprovedCodeAssistant()
    
    @app.route('/complete', methods=['POST'])
    def complete_code():
        try:
            data = request.json
            code = data.get('code', '')
            max_tokens = data.get('max_tokens', 100)
            temperature = data.get('temperature', 0.7)
            
            completion = assistant.complete_code(
                code, 
                max_new_tokens=max_tokens, 
                temperature=temperature
            )
            
            return jsonify({
                'success': True,
                'completion': completion
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/options', methods=['POST'])
    def get_options():
        try:
            data = request.json
            code = data.get('code', '')
            num_options = data.get('num_options', 3)
            
            options = assistant.generate_multiple_options(code, num_options)
            
            return jsonify({
                'success': True,
                'options': options
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app

def create_cli_tool():
    """Command line interface"""
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(description='Code Completion CLI')
        parser.add_argument('--code', required=True, help='Code to complete')
        parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
        parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
        parser.add_argument('--options', type=int, default=1, help='Number of completion options')
        
        args = parser.parse_args()
        
        assistant = ImprovedCodeAssistant()
        
        if args.options > 1:
            completions = assistant.generate_multiple_options(args.code, args.options)
            for i, completion in enumerate(completions, 1):
                print(f"\\n--- Option {i} ---")
                print(completion)
        else:
            completion = assistant.complete_code(
                args.code, 
                max_new_tokens=args.max_tokens, 
                temperature=args.temperature
            )
            print(completion)
    
    return main

if __name__ == "__main__":
    demo_improved_assistant()
