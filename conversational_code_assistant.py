"""
CONVERSATIONAL CODE ASSISTANT
Ask questions in natural English and get Python code as answers!

Examples:
- "Write a function to calculate factorial"
- "Create a class for a bank account"  
- "How do I sort a list?"
- "Make a function to check if a number is prime"
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

class ConversationalCodeAssistant:
    def __init__(self):
        """Load your fine-tuned model for conversational code generation"""
        print("🚀 Loading your fine-tuned CodeT5+ model...")
        print("🗣️  Now with conversational interface!")
        
        self.model_path = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model loaded successfully on {self.device}")
        
        # Define patterns to convert natural language to code prompts
        self.patterns = {
            # Function patterns
            r"write.*function.*to (.+)": "def function_name():",
            r"create.*function.*(.+)": "def function_name():",
            r"make.*function.*(.+)": "def function_name():",
            r"function.*to (.+)": "def function_name():",
            r"function.*for (.+)": "def function_name():",
            
            # Class patterns  
            r"create.*class.*(.+)": "class ClassName:",
            r"write.*class.*(.+)": "class ClassName:",
            r"make.*class.*(.+)": "class ClassName:",
            r"class.*for (.+)": "class ClassName:",
            
            # Algorithm patterns
            r"how.*sort.*list": "def sort_list(arr):",
            r"how.*sort.*array": "def sort_array(arr):",
            r"sorting.*algorithm": "def sort_algorithm(arr):",
            r"bubble.*sort": "def bubble_sort(arr):",
            r"merge.*sort": "def merge_sort(arr):",
            r"quick.*sort": "def quick_sort(arr):",
            
            # Math patterns
            r"calculate.*factorial": "def factorial(n):",
            r"factorial.*function": "def factorial(n):",
            r"fibonacci.*sequence": "def fibonacci(n):",
            r"prime.*number": "def is_prime(n):",
            r"check.*prime": "def is_prime(n):",
            
            # Data structure patterns
            r"linked.*list": "class LinkedList:",
            r"binary.*tree": "class BinaryTree:",
            r"stack.*implementation": "class Stack:",
            r"queue.*implementation": "class Queue:",
            
            # Web/API patterns
            r"web.*scraping": "import requests\\ndef scrape_website(url):",
            r"api.*request": "import requests\\ndef make_api_request(url):",
            r"http.*request": "import requests\\ndef http_request(url):",
            
            # File operations
            r"read.*file": "def read_file(filename):",
            r"write.*file": "def write_file(filename, content):",
            r"file.*handling": "def handle_file(filename):",
            
            # Database patterns
            r"database.*connection": "import sqlite3\\ndef connect_database():",
            r"sql.*query": "def execute_query(query):",
        }
    
    def convert_question_to_code_prompt(self, question):
        """Convert natural language question to code prompt"""
        question_lower = question.lower().strip()
        
        # Check for specific patterns
        for pattern, code_template in self.patterns.items():
            if re.search(pattern, question_lower):
                return code_template
        
        # Extract function name from common patterns
        function_patterns = [
            r"write.*function.*called (\\w+)",
            r"create.*function.*named (\\w+)", 
            r"make.*function.*(\\w+)",
            r"function.*(\\w+)"
        ]
        
        for pattern in function_patterns:
            match = re.search(pattern, question_lower)
            if match:
                func_name = match.group(1)
                return f"def {func_name}():"
        
        # Extract class name
        class_patterns = [
            r"class.*called (\\w+)",
            r"class.*named (\\w+)",
            r"create.*class.*(\\w+)"
        ]
        
        for pattern in class_patterns:
            match = re.search(pattern, question_lower)
            if match:
                class_name = match.group(1).title()
                return f"class {class_name}:"
        
        # Default fallbacks based on keywords
        if any(word in question_lower for word in ['function', 'def', 'method']):
            return "def function_name():"
        elif any(word in question_lower for word in ['class', 'object']):
            return "class ClassName:"
        elif any(word in question_lower for word in ['loop', 'iterate', 'for']):
            return "for i in range(10):"
        elif any(word in question_lower for word in ['if', 'condition', 'check']):
            return "if condition:"
        else:
            # Generic code generation
            return f"# {question}\\ndef solution():"
    
    def generate_code_from_question(self, question, max_tokens=150):
        """Generate code based on natural language question"""
        # Convert question to code prompt
        code_prompt = self.convert_question_to_code_prompt(question)
        
        # Add context to make it more specific
        full_prompt = f"Complete this code:\\n{code_prompt}\\n# {question}"
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=400,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the output
        if "Complete this code:" in result:
            result = result.split("Complete this code:")[-1].strip()
        
        # Remove the original prompt if it appears
        lines = result.split('\\n')
        if lines and code_prompt.strip() in lines[0]:
            result = '\\n'.join(lines[1:])
        
        return result.strip()
    
    def interactive_session(self):
        """Start interactive conversational session"""
        print("\\n" + "="*70)
        print("🤖 CONVERSATIONAL CODE ASSISTANT")
        print("="*70)
        print("Ask me to write code in natural English!")
        print("\\n📝 Examples:")
        print("  • 'Write a function to calculate factorial'")
        print("  • 'Create a class for a bank account'")
        print("  • 'How do I sort a list?'")
        print("  • 'Make a function to check if a number is prime'")
        print("  • 'Function to read a file'")
        print("\\nType 'quit' to exit")
        print("="*70)
        
        while True:
            try:
                # Get user question
                question = input("\\n🗣️  Ask: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("👋 Goodbye! Happy coding!")
                    break
                
                if not question:
                    continue
                
                print("🤖 Thinking...")
                
                # Generate code
                code = self.generate_code_from_question(question)
                
                print("\\n✨ Here's the code:")
                print("-" * 50)
                print(code)
                print("-" * 50)
                
                # Ask if they want variations
                more = input("\\n🔄 Want a different approach? (y/n): ").strip().lower()
                if more in ['y', 'yes']:
                    print("🤖 Generating alternative...")
                    alternative = self.generate_code_from_question(question, max_tokens=120)
                    print("\\n✨ Alternative approach:")
                    print("-" * 50)
                    print(alternative)
                    print("-" * 50)
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye! Happy coding!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Try rephrasing your question or being more specific")

def demo_examples():
    """Show some demo examples"""
    assistant = ConversationalCodeAssistant()
    
    examples = [
        "Write a function to calculate factorial",
        "Create a class for a bank account", 
        "How do I sort a list?",
        "Make a function to check if a number is prime",
        "Function to read a file and count lines",
        "Create a simple calculator class"
    ]
    
    print("\\n" + "="*70)
    print("🎯 DEMO: Natural Language to Code")
    print("="*70)
    
    for i, example in enumerate(examples, 1):
        print(f"\\n📝 Example {i}: '{example}'")
        print("-" * 50)
        
        code = assistant.generate_code_from_question(example)
        print(code)
        print("-" * 50)
        
        input("Press Enter for next example...")

def main():
    """Main function"""
    print("Choose an option:")
    print("1. Interactive conversation mode")
    print("2. See demo examples")
    
    choice = input("\\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        demo_examples()
    else:
        assistant = ConversationalCodeAssistant()
        assistant.interactive_session()

if __name__ == "__main__":
    main()
