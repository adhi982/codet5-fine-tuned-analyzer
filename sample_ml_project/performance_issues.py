
import time
import requests

def inefficient_string_concat(items):
    """Inefficient string concatenation in loop"""
    result = ""
    for item in items:
        result += str(item) + ", "  # Inefficient string concatenation
    return result

def inefficient_loop(data):
    """Inefficient loop using range(len())"""
    results = []
    for i in range(len(data)):  # Should use enumerate
        if data[i] > 0:
            results.append(data[i] * 2)
    return results

def nested_loops_issue(matrix):
    """Nested loops with potential performance issues"""
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):  # Triple nested loop
                total += matrix[i][j][k]
    return total

def inefficient_api_calls(urls):
    """Sequential API calls without async"""
    results = []
    for url in urls:
        response = requests.get(url)  # Should use async or session
        results.append(response.json())
    return results

class SlowProcessor:
    def __init__(self):
        self.cache = {}  # Could use better caching strategy
    
    def process_data(self, data):
        # Recalculating same values
        processed = []
        for item in data:
            # Expensive operation in loop
            result = sum(range(1000)) * item  # Could be cached
            processed.append(result)
        return processed
