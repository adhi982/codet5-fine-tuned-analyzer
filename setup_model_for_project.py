"""
QUICK SETUP SCRIPT
Run this to automatically copy your model to a new project!
"""

import os
import shutil
import sys

def copy_model_to_project(target_project_path):
    """Copy the fine-tuned model to a new project"""
    
    # Source model path
    source_model = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
    
    # Target paths
    target_models_dir = os.path.join(target_project_path, "models")
    target_model = os.path.join(target_models_dir, "codet5p-finetuned")
    
    print(f"🚀 Setting up your fine-tuned model in: {target_project_path}")
    
    # Check if source exists
    if not os.path.exists(source_model):
        print(f"❌ Source model not found at: {source_model}")
        return False
    
    # Create target directory
    try:
        os.makedirs(target_models_dir, exist_ok=True)
        print(f"✅ Created models directory: {target_models_dir}")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return False
    
    # Copy model
    try:
        if os.path.exists(target_model):
            print(f"⚠️  Model already exists at {target_model}")
            response = input("Overwrite? (y/n): ").lower()
            if response != 'y':
                print("❌ Cancelled")
                return False
            shutil.rmtree(target_model)
        
        print("📦 Copying model files...")
        shutil.copytree(source_model, target_model)
        print(f"✅ Model copied to: {target_model}")
        
    except Exception as e:
        print(f"❌ Error copying model: {e}")
        return False
    
    # Copy the portable assistant
    try:
        assistant_source = "E:/Intel Fest/Fine Tune/portable_code_assistant.py"
        assistant_target = os.path.join(target_project_path, "code_assistant.py")
        
        if os.path.exists(assistant_source):
            shutil.copy2(assistant_source, assistant_target)
            print(f"✅ Copied code assistant to: {assistant_target}")
    except Exception as e:
        print(f"⚠️  Could not copy assistant file: {e}")
    
    # Create requirements.txt
    try:
        requirements_path = os.path.join(target_project_path, "requirements.txt")
        requirements_content = """torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
flask>=2.0.0
"""
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        print(f"✅ Created requirements.txt: {requirements_path}")
    except Exception as e:
        print(f"⚠️  Could not create requirements.txt: {e}")
    
    # Create example usage file
    try:
        example_path = os.path.join(target_project_path, "example_usage.py")
        example_content = '''"""
Example usage of your fine-tuned model in this project
"""

from code_assistant import PortableCodeAssistant

def main():
    # Initialize your model
    assistant = PortableCodeAssistant()
    
    # Example 1: Code completion
    code = "def fibonacci(n):"
    completion = assistant.complete_code(code)
    print(f"Input: {code}")
    print(f"Output: {completion}")
    
    # Example 2: Question answering
    question = "How to add two numbers?"
    answer = assistant.answer_question(question)
    print(f"\\nQuestion: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
'''
        with open(example_path, 'w') as f:
            f.write(example_content)
        print(f"✅ Created example usage: {example_path}")
    except Exception as e:
        print(f"⚠️  Could not create example file: {e}")
    
    print("\\n🎉 Setup complete!")
    print("\\n📋 Next steps:")
    print(f"1. cd '{target_project_path}'")
    print("2. pip install -r requirements.txt")
    print("3. python example_usage.py")
    print("\\n🚀 Your model is ready to use!")
    
    return True

def main():
    print("🎯 MODEL COPY SETUP TOOL")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = input("Enter target project path: ").strip()
    
    if not target_path:
        print("❌ No target path provided")
        return
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
            print(f"✅ Created project directory: {target_path}")
        except Exception as e:
            print(f"❌ Could not create directory: {e}")
            return
    
    success = copy_model_to_project(target_path)
    
    if success:
        print("\\n🎉 Success! Your model is now ready in the new project!")
    else:
        print("\\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
