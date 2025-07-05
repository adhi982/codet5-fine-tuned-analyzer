
import subprocess
import pickle
import os

# Security vulnerabilities for testing
PASSWORD = "admin123"  # Hardcoded password
API_KEY = "sk-1234567890abcdef"  # Hardcoded API key

def unsafe_command(user_input):
    """Function with shell injection vulnerability"""
    # Shell injection vulnerability
    result = subprocess.call(f"ls {user_input}", shell=True)
    return result

def unsafe_deserialization(data):
    """Function with pickle vulnerability"""
    # Unsafe pickle deserialization
    return pickle.loads(data)

def unsafe_eval(expression):
    """Function with code injection vulnerability"""
    # Code injection via eval
    return eval(expression)

class DatabaseConnection:
    def __init__(self):
        self.password = "db_password_123"  # Another hardcoded secret
    
    def connect(self, query):
        # SQL injection vulnerability (simulated)
        sql = f"SELECT * FROM users WHERE name = '{query}'"
        return sql
