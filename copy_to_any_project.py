"""
MINIMAL INTEGRATION: Copy this to any Python project
Just change the model_path to your model location
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SimpleCodeAssistant:
    def __init__(self, model_path="E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"):
        """Load your fine-tuned model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def complete(self, code):
        """Complete the given code snippet"""
        prompt = f"Complete this code:\n{code}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200, num_beams=3, temperature=0.8)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("Complete this code:")[-1].strip()

# ===============================================
# USAGE EXAMPLES - Copy any of these patterns
# ===============================================

def example_1_basic_usage():
    """Basic code completion"""
    assistant = SimpleCodeAssistant()
    
    code = "def fibonacci(n):"
    completion = assistant.complete(code)
    print(f"Input: {code}")
    print(f"Output: {completion}")

def example_2_web_api():
    """Flask API wrapper"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    assistant = SimpleCodeAssistant()
    
    @app.route('/complete', methods=['POST'])
    def complete_code():
        data = request.json
        code = data.get('code', '')
        completion = assistant.complete(code)
        return jsonify({'completion': completion})
    
    # Run with: app.run(host='0.0.0.0', port=5000)

def example_3_command_line():
    """Command line tool"""
    import sys
    
    if len(sys.argv) > 1:
        code = ' '.join(sys.argv[1:])
        assistant = SimpleCodeAssistant()
        completion = assistant.complete(code)
        print(completion)

def example_4_file_processing():
    """Process files with code completion"""
    assistant = SimpleCodeAssistant()
    
    def complete_file(input_file, output_file):
        with open(input_file, 'r') as f:
            code = f.read()
        
        completion = assistant.complete(code)
        
        with open(output_file, 'w') as f:
            f.write(completion)

# ===============================================
# INTEGRATION IN YOUR EXISTING PROJECT
# ===============================================

"""
To use in your existing project:

1. Copy the SimpleCodeAssistant class above
2. Install requirements: pip install torch transformers
3. Use like this:

    # In your project file
    from your_file import SimpleCodeAssistant
    
    assistant = SimpleCodeAssistant()
    completed_code = assistant.complete("def your_function():")
    print(completed_code)
"""

if __name__ == "__main__":
    # Run the basic example
    example_1_basic_usage()
