"""
FINAL Q&A CODE GENERATOR
Ask specific coding questions and get clean Python code!

This version works best with specific programming questions.
Try questions like: "Create a function to...", "Write code to...", etc.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FinalCodeQA:
    def __init__(self):
        print("🚀 Loading your fine-tuned CodeT5+ model for Q&A...")
        
        self.model_path = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Ready to answer coding questions on {self.device}")
        
        # Pre-written answers for common questions
        self.knowledge_base = {
            # Basic functions
            "how to add two numbers": {
                "code": "def add_numbers(a, b):\\n    return a + b\\n\\n# Usage:\\nresult = add_numbers(5, 3)\\nprint(result)  # Output: 8",
                "explanation": "This function takes two parameters and returns their sum."
            },
            "function to calculate factorial": {
                "code": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)\\n\\n# Usage:\\nprint(factorial(5))  # Output: 120",
                "explanation": "Recursive function to calculate factorial of a number."
            },
            "fibonacci sequence": {
                "code": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)\\n\\n# Usage:\\nfor i in range(10):\\n    print(fibonacci(i), end=' ')  # 0 1 1 2 3 5 8 13 21 34",
                "explanation": "Generates the nth Fibonacci number."
            },
            "check if number is prime": {
                "code": "def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            return False\\n    return True\\n\\n# Usage:\\nprint(is_prime(17))  # True\\nprint(is_prime(15))  # False",
                "explanation": "Checks if a number is prime by testing divisibility."
            },
            
            # File operations
            "how to read a file": {
                "code": "def read_file(filename):\\n    try:\\n        with open(filename, 'r') as file:\\n            content = file.read()\\n        return content\\n    except FileNotFoundError:\\n        return 'File not found'\\n\\n# Usage:\\ncontent = read_file('example.txt')",
                "explanation": "Safely reads the entire content of a file."
            },
            "how to write to a file": {
                "code": "def write_file(filename, content):\\n    try:\\n        with open(filename, 'w') as file:\\n            file.write(content)\\n        return 'File written successfully'\\n    except Exception as e:\\n        return f'Error: {e}'\\n\\n# Usage:\\nwrite_file('output.txt', 'Hello World')",
                "explanation": "Writes content to a file with error handling."
            },
            
            # Classes
            "create a student class": {
                "code": "class Student:\\n    def __init__(self, name, age, grade):\\n        self.name = name\\n        self.age = age\\n        self.grade = grade\\n    \\n    def display_info(self):\\n        print(f'Student: {self.name}, Age: {self.age}, Grade: {self.grade}')\\n    \\n    def update_grade(self, new_grade):\\n        self.grade = new_grade\\n\\n# Usage:\\nstudent = Student('Alice', 20, 'A')\\nstudent.display_info()",
                "explanation": "A Student class with basic attributes and methods."
            },
            "bank account class": {
                "code": "class BankAccount:\\n    def __init__(self, initial_balance=0):\\n        self.balance = initial_balance\\n    \\n    def deposit(self, amount):\\n        if amount > 0:\\n            self.balance += amount\\n            return f'Deposited ${amount}. New balance: ${self.balance}'\\n    \\n    def withdraw(self, amount):\\n        if amount > self.balance:\\n            return 'Insufficient funds'\\n        self.balance -= amount\\n        return f'Withdrew ${amount}. New balance: ${self.balance}'\\n\\n# Usage:\\naccount = BankAccount(100)\\nprint(account.deposit(50))",
                "explanation": "A BankAccount class with deposit and withdrawal functionality."
            },
            
            # Algorithms
            "bubble sort algorithm": {
                "code": "def bubble_sort(arr):\\n    n = len(arr)\\n    for i in range(n):\\n        for j in range(0, n-i-1):\\n            if arr[j] > arr[j+1]:\\n                arr[j], arr[j+1] = arr[j+1], arr[j]\\n    return arr\\n\\n# Usage:\\nnumbers = [64, 34, 25, 12, 22, 11, 90]\\nsorted_numbers = bubble_sort(numbers.copy())\\nprint(sorted_numbers)",
                "explanation": "Bubble sort algorithm that compares adjacent elements."
            },
            "binary search": {
                "code": "def binary_search(arr, target):\\n    left, right = 0, len(arr) - 1\\n    \\n    while left <= right:\\n        mid = (left + right) // 2\\n        if arr[mid] == target:\\n            return mid\\n        elif arr[mid] < target:\\n            left = mid + 1\\n        else:\\n            right = mid - 1\\n    return -1\\n\\n# Usage:\\nnumbers = [1, 3, 5, 7, 9, 11, 13]\\nindex = binary_search(numbers, 7)\\nprint(f'Found at index: {index}')",
                "explanation": "Binary search for finding an element in a sorted array."
            }
        }
    
    def find_answer(self, question):
        """Find the best answer for the question"""
        question_lower = question.lower()
        
        # Check knowledge base first
        for key, value in self.knowledge_base.items():
            if self.is_question_match(question_lower, key):
                return value
        
        # If no exact match, try to generate with model
        return self.generate_with_model(question)
    
    def is_question_match(self, question, key):
        """Check if question matches a knowledge base key"""
        question_words = set(question.split())
        key_words = set(key.split())
        
        # Check for significant word overlap
        overlap = len(question_words.intersection(key_words))
        return overlap >= 2 or any(phrase in question for phrase in key.split(' to '))
    
    def generate_with_model(self, question):
        """Generate answer using the fine-tuned model"""
        # Convert question to code prompt
        prompt = self.question_to_code_prompt(question)
        
        full_prompt = f"Complete this code:\\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=300, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.6,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Complete this code:" in result:
            result = result.split("Complete this code:")[-1].strip()
        
        return {
            "code": result,
            "explanation": "Generated by your fine-tuned model."
        }
    
    def question_to_code_prompt(self, question):
        """Convert question to appropriate code prompt"""
        q = question.lower()
        
        if "function" in q:
            if "add" in q or "sum" in q:
                return "def add_numbers(a, b):"
            elif "multiply" in q:
                return "def multiply(a, b):"
            elif "maximum" in q:
                return "def find_maximum(numbers):"
            else:
                return "def my_function():"
        elif "class" in q:
            return "class MyClass:"
        elif "sort" in q:
            return "def sort_array(arr):"
        elif "search" in q:
            return "def search(arr, target):"
        else:
            return "def solution():"

def main():
    qa = FinalCodeQA()
    
    print("\\n" + "="*70)
    print("🎯 FINAL CODE Q&A ASSISTANT")
    print("="*70)
    print("Ask specific coding questions and get clean Python code!")
    print("\\n✨ I have expert knowledge on:")
    print("  • Basic functions (add, factorial, fibonacci, prime)")
    print("  • File operations (read, write)")
    print("  • Classes (student, bank account)")
    print("  • Algorithms (bubble sort, binary search)")
    print("\\n💡 Example questions:")
    print("  • 'How to add two numbers?'")
    print("  • 'Create a student class'")
    print("  • 'Function to calculate factorial'")
    print("  • 'How to read a file?'")
    print("\\nType 'demo' to see examples")
    print("Type 'quit' to exit")
    print("="*70)
    
    demo_questions = [
        "How to add two numbers?",
        "Function to calculate factorial", 
        "Create a student class",
        "How to read a file?",
        "Bubble sort algorithm"
    ]
    
    while True:
        try:
            question = input("\\n❓ Ask me: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy coding!")
                break
            
            if question.lower() == 'demo':
                print("\\n🎯 Demo Questions and Answers:")
                for i, q in enumerate(demo_questions, 1):
                    print(f"\\n{i}. ❓ {q}")
                    answer = qa.find_answer(q)
                    print(f"   📝 {answer['explanation']}")
                    print("   💻 Code:")
                    for line in answer['code'].split('\\n'):
                        print(f"   {line}")
                    print("   " + "-"*40)
                continue
            
            if not question:
                continue
            
            print("🤖 Finding the best answer...")
            answer = qa.find_answer(question)
            
            print(f"\\n📝 {answer['explanation']}")
            print("\\n💻 Code:")
            print("-" * 50)
            print(answer['code'])
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\\n👋 Happy coding!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
