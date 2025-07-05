# 🚀 HOW TO USE YOUR FINE-TUNED MODEL IN YOUR PROJECTS

## 📋 QUICK START

Your fine-tuned model is ready! Here are several ways to integrate it into your projects:

## 🔧 METHOD 1: Simple Python Integration

### Step 1: Create a Simple Wrapper
```python
# save as: my_code_assistant.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MyCodeAssistant:
    def __init__(self):
        model_path = "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    
    def complete_code(self, code_snippet):
        prompt = f"Complete this code:\n{code_snippet}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200, num_beams=3)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("Complete this code:")[-1].strip()

# Usage in your project:
assistant = MyCodeAssistant()
completed = assistant.complete_code("def fibonacci(n):")
print(completed)
```

## 🌐 METHOD 2: Flask API Service

### Create a Web Service
```python
# save as: code_completion_api.py
from flask import Flask, request, jsonify
from my_code_assistant import MyCodeAssistant

app = Flask(__name__)
assistant = MyCodeAssistant()

@app.route('/complete', methods=['POST'])
def complete_code():
    data = request.json
    code = data.get('code', '')
    
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    try:
        completed = assistant.complete_code(code)
        return jsonify({'completed_code': completed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Use the API from any project:
```python
import requests

# Send code to complete
response = requests.post('http://localhost:5000/complete', 
                        json={'code': 'def bubble_sort(arr):'})
completed_code = response.json()['completed_code']
print(completed_code)
```

## 🖥️ METHOD 3: VS Code Extension Integration

### Create Extension Script
```javascript
// For VS Code extension
const axios = require('axios');

async function completeCode(codeSnippet) {
    try {
        const response = await axios.post('http://localhost:5000/complete', {
            code: codeSnippet
        });
        return response.data.completed_code;
    } catch (error) {
        console.error('Code completion failed:', error);
        return null;
    }
}

// Usage in VS Code extension
const completedCode = await completeCode('def process_data():');
```

## 🐍 METHOD 4: Jupyter Notebook Integration

### In Jupyter Notebook
```python
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

from my_code_assistant import MyCodeAssistant
assistant = MyCodeAssistant()

# Cell 2: Magic command for code completion
def complete_code_magic(line):
    completed = assistant.complete_code(line)
    print("Completed code:")
    print(completed)

# Usage
complete_code_magic("def merge_sort(arr):")
```

## 🚀 METHOD 5: Command Line Tool

### Create CLI Tool
```python
# save as: code_complete_cli.py
import argparse
from my_code_assistant import MyCodeAssistant

def main():
    parser = argparse.ArgumentParser(description='Complete code using fine-tuned CodeT5+')
    parser.add_argument('--code', type=str, required=True, help='Code snippet to complete')
    parser.add_argument('--length', type=int, default=200, help='Max completion length')
    
    args = parser.parse_args()
    
    assistant = MyCodeAssistant()
    completed = assistant.complete_code(args.code)
    
    print("Original code:")
    print(args.code)
    print("\nCompleted code:")
    print(completed)

if __name__ == '__main__':
    main()
```

### Usage
```bash
python code_complete_cli.py --code "def calculate_average(numbers):"
```

## 📱 METHOD 6: FastAPI Modern Web Service

### Modern API with FastAPI
```python
# save as: fastapi_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from my_code_assistant import MyCodeAssistant

app = FastAPI(title="CodeT5+ Completion Service")
assistant = MyCodeAssistant()

class CodeRequest(BaseModel):
    code: str
    max_length: int = 200

class CodeResponse(BaseModel):
    original_code: str
    completed_code: str

@app.post("/complete", response_model=CodeResponse)
async def complete_code(request: CodeRequest):
    try:
        completed = assistant.complete_code(request.code)
        return CodeResponse(
            original_code=request.code,
            completed_code=completed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn fastapi_service:app --reload
```

## 🔄 METHOD 7: Real-time Code Completion Server

### WebSocket Server for Real-time Completion
```python
# save as: websocket_server.py
import asyncio
import websockets
import json
from my_code_assistant import MyCodeAssistant

assistant = MyCodeAssistant()

async def handle_completion(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            code_snippet = data.get('code', '')
            
            if code_snippet:
                completed = assistant.complete_code(code_snippet)
                response = {
                    'status': 'success',
                    'completed_code': completed
                }
            else:
                response = {'status': 'error', 'message': 'No code provided'}
            
            await websocket.send(json.dumps(response))
        
        except Exception as e:
            error_response = {'status': 'error', 'message': str(e)}
            await websocket.send(json.dumps(error_response))

# Start server
start_server = websockets.serve(handle_completion, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

## 📦 METHOD 8: Package Your Model

### Create a Python Package
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my-codet5-assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    author="Your Name",
    description="Fine-tuned CodeT5+ for code completion",
)
```

### Then install and use anywhere:
```bash
pip install -e .

# In any Python project:
from my_codet5_assistant import MyCodeAssistant
assistant = MyCodeAssistant()
result = assistant.complete_code("def hello_world():")
```

## 🎯 PRACTICAL EXAMPLES

### Example 1: IDE Plugin
```python
# For PyCharm/VS Code plugin
def get_code_suggestion(current_line, cursor_position):
    code_before_cursor = current_line[:cursor_position]
    completed = assistant.complete_code(code_before_cursor)
    return completed
```

### Example 2: Code Review Assistant
```python
def suggest_improvements(code_snippet):
    improved = assistant.complete_code(f"Improve this code:\n{code_snippet}")
    return improved
```

### Example 3: Educational Tool
```python
def explain_and_complete(partial_code):
    completed = assistant.complete_code(partial_code)
    return {
        'original': partial_code,
        'completed': completed,
        'explanation': f"This code completes the function by adding the missing implementation"
    }
```

## 🚀 DEPLOYMENT OPTIONS

1. **Local Development**: Use directly in Python scripts
2. **Web Service**: Deploy as Flask/FastAPI service
3. **Docker Container**: Package with Docker for easy deployment
4. **Cloud Service**: Deploy on AWS/Azure/GCP
5. **Edge Device**: Run on local servers for privacy
6. **IDE Integration**: Create plugins for popular IDEs

## ⚡ PERFORMANCE TIPS

1. **Load Once**: Initialize the model once and reuse
2. **Batch Processing**: Process multiple requests together
3. **Caching**: Cache common completions
4. **GPU Acceleration**: Use CUDA when available
5. **Model Quantization**: Use smaller model variants for faster inference

Your fine-tuned CodeT5+ model is now ready to be integrated into any project! 🎉
