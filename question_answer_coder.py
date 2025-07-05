"""
QUESTION & ANSWER CODE GENERATOR
Ask natural language questions and get Python code answers!

Just ask like:
- "How do I create a function to add two numbers?"
- "Write code to read a file"
- "Show me a class for a student record"
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    """Load the fine-tuned model"""
    print("🚀 Loading your fine-tuned CodeT5+ model...")
    
    model_path = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model ready on {device}")
    return tokenizer, model, device

def question_to_code_starter(question):
    """Convert question to appropriate code starter"""
    q = question.lower()
    
    # Function questions
    if "function" in q and "add" in q and "number" in q:
        return "def add_numbers(a, b):"
    elif "function" in q and "factorial" in q:
        return "def factorial(n):"
    elif "function" in q and "fibonacci" in q:
        return "def fibonacci(n):"
    elif "function" in q and "prime" in q:
        return "def is_prime(n):"
    elif "function" in q and "read" in q and "file" in q:
        return "def read_file(filename):"
    elif "function" in q and "write" in q and "file" in q:
        return "def write_file(filename, content):"
    elif "function" in q and "sort" in q:
        return "def sort_list(arr):"
    elif "function" in q:
        return "def my_function():"
    
    # Class questions
    elif "class" in q and "student" in q:
        return "class Student:"
    elif "class" in q and "bank" in q:
        return "class BankAccount:"
    elif "class" in q and "car" in q:
        return "class Car:"
    elif "class" in q and "person" in q:
        return "class Person:"
    elif "class" in q and "calculator" in q:
        return "class Calculator:"
    elif "class" in q:
        return "class MyClass:"
    
    # Specific algorithms
    elif "bubble sort" in q:
        return "def bubble_sort(arr):"
    elif "binary search" in q:
        return "def binary_search(arr, target):"
    elif "merge sort" in q:
        return "def merge_sort(arr):"
    elif "quick sort" in q:
        return "def quick_sort(arr):"
    
    # File operations
    elif "read file" in q:
        return "with open('filename.txt', 'r') as file:"
    elif "write file" in q:
        return "with open('filename.txt', 'w') as file:"
    
    # Loop patterns
    elif "loop" in q or "iterate" in q:
        return "for i in range(10):"
    elif "while" in q:
        return "while condition:"
    
    # Condition patterns
    elif "if" in q or "condition" in q:
        return "if condition:"
    
    # Web/API
    elif "web" in q or "api" in q or "request" in q:
        return "import requests\\ndef make_request(url):"
    
    # Default
    else:
        return "def solution():"

def answer_question(tokenizer, model, device, question):
    """Generate code answer for the question"""
    # Get appropriate code starter
    code_starter = question_to_code_starter(question)
    
    # Create prompt
    prompt = f"Complete this code:\\n{code_starter}\\n# {question}"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,
            top_p=0.9
        )
    
    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean output
    if "Complete this code:" in result:
        result = result.split("Complete this code:")[-1].strip()
    
    return result

def main():
    # Load model
    tokenizer, model, device = load_model()
    
    print("\\n" + "="*60)
    print("🤖 ASK ME CODING QUESTIONS!")
    print("="*60)
    print("Ask me how to write code in natural English!")
    print("\\n📝 Example questions:")
    print("  • How do I create a function to add two numbers?")
    print("  • Write code to read a file")
    print("  • Show me a class for a student record")
    print("  • How to sort a list in Python?")
    print("  • Create a function to check if number is prime")
    print("\\nType 'quit' to exit")
    print("="*60)
    
    # Example questions for demo
    demo_questions = [
        "How do I create a function to add two numbers?",
        "Write code to read a file",
        "Show me a class for a student record", 
        "How to check if a number is prime?",
        "Create a simple calculator class"
    ]
    
    print("\\n🎯 QUICK DEMO - Here are some example answers:")
    for i, q in enumerate(demo_questions[:3], 1):
        print(f"\\n❓ Question {i}: {q}")
        answer = answer_question(tokenizer, model, device, q)
        print(f"💻 Answer:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
    
    print("\\n🗣️  Now ask your own questions!")
    
    while True:
        try:
            question = input("\\n❓ Your Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy coding!")
                break
            
            if not question:
                continue
            
            print("🤖 Generating answer...")
            answer = answer_question(tokenizer, model, device, question)
            
            print("\\n💻 Answer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\\n👋 Happy coding!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
