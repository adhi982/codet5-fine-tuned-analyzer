"""
PORTABLE CODE ASSISTANT
Copy this file to any project and use your fine-tuned model!

SETUP:
1. Copy your model folder: E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned/
2. Put it in your project directory
3. Update the model_path below
4. Install: pip install torch transformers accelerate
5. Use the assistant!
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class PortableCodeAssistant:
    """
    Portable version of your fine-tuned CodeT5+ model
    Copy this class to any project!
    """
    
    def __init__(self, model_path=None):
        """
        Initialize with your fine-tuned model
        
        Args:
            model_path: Path to your codet5p-finetuned folder
                       If None, will look in common locations
        """
        if model_path is None:
            # Auto-detect model path in common locations
            possible_paths = [
                "./models/codet5p-finetuned",
                "./codet5p-finetuned", 
                "../models/codet5p-finetuned",
                "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"  # Original location
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "Model not found! Please copy your model folder and update the path."
                )
        
        print(f"🚀 Loading model from: {model_path}")
        
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
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure you've copied the model folder and installed dependencies")
            raise
    
    def complete_code(self, code, max_tokens=100, temperature=0.7):
        """
        Complete the given code snippet
        
        Args:
            code: Incomplete code to complete
            max_tokens: Maximum tokens to generate
            temperature: Generation randomness (0.1-1.0)
        
        Returns:
            Completed code as string
        """
        prompt = f"Complete this code:\\n{code}"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=400, 
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Complete this code:" in result:
            result = result.split("Complete this code:")[-1].strip()
        
        return result
    
    def answer_question(self, question):
        """
        Answer coding questions with code
        
        Args:
            question: Natural language question
            
        Returns:
            Code answer
        """
        # Simple question to code conversion
        if "function" in question.lower():
            if "add" in question.lower():
                prompt = "def add_numbers(a, b):"
            elif "factorial" in question.lower():
                prompt = "def factorial(n):"
            elif "fibonacci" in question.lower():
                prompt = "def fibonacci(n):"
            else:
                prompt = "def my_function():"
        elif "class" in question.lower():
            prompt = "class MyClass:"
        else:
            prompt = f"# {question}\\ndef solution():"
        
        return self.complete_code(prompt)
    
    def generate_multiple(self, code, num_options=3):
        """Generate multiple completion options"""
        options = []
        for i in range(num_options):
            temp = 0.5 + (i * 0.2)  # Different temperatures
            completion = self.complete_code(code, temperature=temp)
            if completion not in options:
                options.append(completion)
        return options

# ================================
# USAGE EXAMPLES
# ================================

def example_basic_usage():
    """Basic usage example"""
    print("🎯 Example 1: Basic Code Completion")
    
    assistant = PortableCodeAssistant()
    
    code = "def fibonacci(n):"
    completion = assistant.complete_code(code)
    
    print(f"Input: {code}")
    print(f"Output: {completion}")

def example_question_answering():
    """Question answering example"""
    print("\\n🎯 Example 2: Question Answering")
    
    assistant = PortableCodeAssistant()
    
    questions = [
        "How to add two numbers?",
        "Function to calculate factorial",
        "Create a simple class"
    ]
    
    for question in questions:
        answer = assistant.answer_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}\\n")

def example_web_integration():
    """Web app integration example"""
    print("🎯 Example 3: Web App Integration")
    
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        assistant = PortableCodeAssistant()
        
        @app.route('/complete', methods=['POST'])
        def complete():
            data = request.json
            code = data.get('code', '')
            completion = assistant.complete_code(code)
            return jsonify({'completion': completion})
        
        print("Flask app ready! Use: app.run()")
        
    except ImportError:
        print("Flask not installed. Install with: pip install flask")

def interactive_demo():
    """Interactive demo"""
    assistant = PortableCodeAssistant()
    
    print("\\n" + "="*50)
    print("🤖 INTERACTIVE CODE ASSISTANT")
    print("="*50)
    print("Type code to complete or ask questions!")
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\\n💻 Enter code or question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Detect if it's a question or code
            if any(word in user_input.lower() for word in ['how', 'what', 'create', 'write', '?']):
                result = assistant.answer_question(user_input)
                print(f"🤖 Answer: {result}")
            else:
                result = assistant.complete_code(user_input)
                print(f"✨ Completion: {result}")
                
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# ================================
# INTEGRATION TEMPLATES
# ================================

class FlaskApp:
    """Ready-to-use Flask app template"""
    
    def __init__(self):
        try:
            from flask import Flask, request, jsonify
            self.Flask = Flask
            self.request = request 
            self.jsonify = jsonify
        except ImportError:
            print("Install Flask: pip install flask")
            return
        
        self.app = Flask(__name__)
        self.assistant = PortableCodeAssistant()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/complete', methods=['POST'])
        def complete():
            data = self.request.json
            code = data.get('code', '')
            completion = self.assistant.complete_code(code)
            return self.jsonify({'completion': completion})
        
        @self.app.route('/question', methods=['POST']) 
        def question():
            data = self.request.json
            question = data.get('question', '')
            answer = self.assistant.answer_question(question)
            return self.jsonify({'answer': answer})
    
    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)

class CommandLineTool:
    """Ready-to-use CLI template"""
    
    def __init__(self):
        self.assistant = PortableCodeAssistant()
    
    def run(self, args):
        import sys
        
        if len(args) < 2:
            print("Usage: python script.py 'code to complete'")
            return
        
        code = ' '.join(args[1:])
        completion = self.assistant.complete_code(code)
        print(completion)

# ================================
# MAIN EXECUTION
# ================================

def main():
    """Main function - choose what to run"""
    print("🚀 PORTABLE CODE ASSISTANT")
    print("Choose an option:")
    print("1. Interactive demo")
    print("2. Basic usage example")
    print("3. Question answering example")
    print("4. Start Flask web app")
    
    choice = input("\\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        example_basic_usage()
    elif choice == "3":
        example_question_answering() 
    elif choice == "4":
        flask_app = FlaskApp()
        print("Starting Flask app on http://localhost:5000")
        flask_app.run()
    else:
        print("Running interactive demo by default...")
        interactive_demo()

if __name__ == "__main__":
    main()
