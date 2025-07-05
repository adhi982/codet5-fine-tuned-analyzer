"""
QUICK DEMO: Using Your Fine-Tuned CodeT5+ Model
Run this to see your model in action!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ready_to_use_assistant import CodeT5Assistant

def main():
    print("=" * 60)
    print("🎯 DEMO: Your Fine-Tuned CodeT5+ Model in Action")
    print("=" * 60)
    
    # Initialize your model
    assistant = CodeT5Assistant()
    
    # Example 1: Complete a function
    print("\n📝 Example 1: Function Completion")
    print("-" * 40)
    code_snippet = "def calculate_factorial(n):"
    print(f"Input: {code_snippet}")
    completion = assistant.complete_code(code_snippet)
    print(f"Output:\n{completion}")
    
    # Example 2: Complete a class
    print("\n📝 Example 2: Class Completion")
    print("-" * 40)
    class_snippet = "class BankAccount:\n    def __init__(self, balance):"
    print(f"Input: {class_snippet}")
    completion = assistant.complete_code(class_snippet)
    print(f"Output:\n{completion}")
    
    # Example 3: Algorithm implementation
    print("\n📝 Example 3: Algorithm Implementation")
    print("-" * 40)
    algo_snippet = "def binary_search(arr, target):"
    print(f"Input: {algo_snippet}")
    completion = assistant.complete_code(algo_snippet)
    print(f"Output:\n{completion}")
    
    # Example 4: Interactive mode
    print("\n🎮 Interactive Mode")
    print("-" * 40)
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\n💻 Enter code to complete: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                completion = assistant.complete_code(user_input)
                print(f"✨ Completion:\n{completion}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
