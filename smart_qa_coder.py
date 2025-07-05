"""
IMPROVED Q&A CODE GENERATOR
Ask natural English questions and get clean Python code answers!

This version focuses on common programming questions and provides cleaner outputs.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SmartCodeQA:
    def __init__(self):
        """Initialize the Q&A code generator"""
        print("🚀 Loading your fine-tuned CodeT5+ model...")
        
        self.model_path = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model ready on {self.device}")
        
        # Predefined Q&A pairs for common questions
        self.qa_templates = {
            "how to add two numbers": "def add_numbers(a, b):\\n    return a + b",
            "function to add numbers": "def add_numbers(a, b):\\n    return a + b",
            "calculate factorial": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
            "check prime number": "def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            return False\\n    return True",
            "fibonacci sequence": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
            "read file": "def read_file(filename):\\n    with open(filename, 'r') as file:\\n        content = file.read()\\n    return content",
            "write file": "def write_file(filename, content):\\n    with open(filename, 'w') as file:\\n        file.write(content)",
            "sort list": "def sort_list(arr):\\n    return sorted(arr)",
            "bubble sort": "def bubble_sort(arr):\\n    n = len(arr)\\n    for i in range(n):\\n        for j in range(0, n-i-1):\\n            if arr[j] > arr[j+1]:\\n                arr[j], arr[j+1] = arr[j+1], arr[j]\\n    return arr",
            "student class": "class Student:\\n    def __init__(self, name, age, grade):\\n        self.name = name\\n        self.age = age\\n        self.grade = grade\\n    \\n    def display_info(self):\\n        print(f'Name: {self.name}, Age: {self.age}, Grade: {self.grade}')",
            "bank account class": "class BankAccount:\\n    def __init__(self, balance=0):\\n        self.balance = balance\\n    \\n    def deposit(self, amount):\\n        self.balance += amount\\n    \\n    def withdraw(self, amount):\\n        if amount <= self.balance:\\n            self.balance -= amount\\n        else:\\n            print('Insufficient funds')",
            "calculator class": "class Calculator:\\n    def add(self, a, b):\\n        return a + b\\n    \\n    def subtract(self, a, b):\\n        return a - b\\n    \\n    def multiply(self, a, b):\\n        return a * b\\n    \\n    def divide(self, a, b):\\n        if b != 0:\\n            return a / b\\n        return 'Cannot divide by zero'",
        }
    
    def find_template_match(self, question):
        """Find if question matches any template"""
        question_lower = question.lower()
        
        for key, template in self.qa_templates.items():
            if key in question_lower:
                return template
        return None
    
    def generate_code_answer(self, question):
        """Generate code answer for the question"""
        # First check templates
        template_match = self.find_template_match(question)
        if template_match:
            return template_match
        
        # If no template match, use the model
        # Convert question to a code prompt
        code_prompt = self.question_to_prompt(question)
        
        # Generate using model
        prompt = f"Complete this code:\\n{code_prompt}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=300, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.6,
                do_sample=True,
                repetition_penalty=1.3,
                top_p=0.8
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean output
        if "Complete this code:" in result:
            result = result.split("Complete this code:")[-1].strip()
        
        return result
    
    def question_to_prompt(self, question):
        """Convert question to appropriate code prompt"""
        q = question.lower()
        
        if "function" in q:
            if "add" in q or "sum" in q:
                return "def add_numbers(a, b):"
            elif "multiply" in q:
                return "def multiply(a, b):"
            elif "divide" in q:
                return "def divide(a, b):"
            elif "maximum" in q or "max" in q:
                return "def find_max(numbers):"
            elif "minimum" in q or "min" in q:
                return "def find_min(numbers):"
            else:
                return "def my_function():"
        
        elif "class" in q:
            if "person" in q:
                return "class Person:"
            elif "car" in q:
                return "class Car:"
            else:
                return "class MyClass:"
        
        elif "loop" in q or "iterate" in q:
            return "for i in range(10):"
        
        elif "list" in q and "sort" in q:
            return "def sort_list(arr):"
        
        else:
            return "def solution():"

def main():
    """Main interactive function"""
    qa = SmartCodeQA()
    
    print("\\n" + "="*70)
    print("🤖 SMART CODE Q&A ASSISTANT")
    print("="*70)
    print("Ask me coding questions in plain English!")
    print("\\n🎯 I'm great at these topics:")
    print("  ✓ Basic functions (add, multiply, factorial, fibonacci)")
    print("  ✓ Classes (student, bank account, calculator)")
    print("  ✓ File operations (read, write)")
    print("  ✓ Algorithms (sorting, searching)")
    print("  ✓ Data structures (lists, loops)")
    print("\\nType 'examples' to see demo questions")
    print("Type 'quit' to exit")
    print("="*70)
    
    # Common questions for quick demo
    common_questions = [
        "How to add two numbers?",
        "Create a function to calculate factorial", 
        "Show me a student class",
        "How to read a file?",
        "Function to check if number is prime",
        "Create a simple calculator class",
        "How to sort a list?",
        "Write a bubble sort function"
    ]
    
    while True:
        try:
            question = input("\\n❓ Your Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy coding!")
                break
            
            if question.lower() == 'examples':
                print("\\n🎯 Here are some example questions and answers:")
                for i, q in enumerate(common_questions[:4], 1):
                    print(f"\\n{i}. ❓ {q}")
                    answer = qa.generate_code_answer(q)
                    print("   💻 Answer:")
                    print("   " + "-"*40)
                    for line in answer.split('\\n'):
                        print(f"   {line}")
                    print("   " + "-"*40)
                continue
            
            if not question:
                continue
            
            print("🤖 Generating code...")
            answer = qa.generate_code_answer(question)
            
            print("\\n💻 Here's your code:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
            # Ask for more help
            more = input("\\n🔄 Need a different approach or have another question? (y/n): ").lower()
            if more in ['y', 'yes']:
                follow_up = input("What specifically would you like to know? ")
                if follow_up.strip():
                    answer2 = qa.generate_code_answer(follow_up)
                    print("\\n💻 Alternative solution:")
                    print("-" * 50)
                    print(answer2)
                    print("-" * 50)
                    
        except KeyboardInterrupt:
            print("\\n👋 Happy coding!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try asking about basic functions, classes, or algorithms")

if __name__ == "__main__":
    main()
