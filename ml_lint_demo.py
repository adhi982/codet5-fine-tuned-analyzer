"""
ML LINT AGENT DEMO
Comprehensive demonstration of the ML-Integrated Lint Agent

This script shows all the features:
- Intelligent code review
- Learning from feedback
- Personalized suggestions
- Security analysis
- Performance optimization
- Natural language explanations
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_lint_agent import MLLintAgent, CodeIssue, DeveloperFeedback

def create_sample_code_files():
    """Create sample Python files with various issues for testing"""
    
    # Create samples directory
    samples_dir = Path("sample_code")
    samples_dir.mkdir(exist_ok=True)
    
    # Sample 1: Security issues
    security_sample = '''
import subprocess
import pickle
import os

# Security vulnerabilities
password = "admin123"  # Hardcoded password
api_key = "sk-1234567890abcdef"  # Hardcoded API key

def run_command(user_input):
    # Shell injection vulnerability
    subprocess.call(f"ls {user_input}", shell=True)

def load_data(file_path):
    # Unsafe pickle deserialization
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def execute_query(query, param):
    # SQL injection vulnerability
    cursor.execute(f"SELECT * FROM users WHERE name = '{param}'")
    return cursor.fetchall()
'''
    
    with open(samples_dir / "security_issues.py", "w") as f:
        f.write(security_sample)
    
    # Sample 2: Performance issues
    performance_sample = '''
import time

def inefficient_loop(items):
    # Should use enumerate instead
    for i in range(len(items)):
        print(f"Item {i}: {items[i]}")

def string_concatenation(words):
    # Inefficient string concatenation
    result = ""
    for word in words:
        result += word + " "
    return result

def nested_loops(matrix):
    # O(n^3) complexity - could be optimized
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix)):
                if matrix[i][j] == matrix[k][j]:
                    print(f"Match found at {i},{j} and {k},{j}")

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_items(self, items):
        # Inefficient list operations
        for item in items:
            self.data.append(item)  # Should use extend for bulk operations
            
    def find_item(self, target):
        # Linear search in potentially large list
        for item in self.data:
            if item.id == target:
                return item
        return None
'''
    
    with open(samples_dir / "performance_issues.py", "w") as f:
        f.write(performance_sample)
    
    # Sample 3: Code quality issues
    quality_sample = '''
def complex_function(a, b, c, d, e, f, g, h):
    # Too many parameters
    if a and b and c and d and e and f and g and h:
        # Complex condition
        try:
            result = a + b * c - d / e + f ** g % h
            if result > 100:
                if result > 200:
                    if result > 300:
                        # Deeply nested conditions
                        return "very high"
                    else:
                        return "high"
                else:
                    return "medium"
            else:
                return "low"
        except:
            # Empty except clause
            pass
    
    # Long function continues...
    for i in range(100):
        for j in range(100):
            for k in range(100):
                # Deeply nested loops
                if i == j == k:
                    print(f"Triple match: {i}")

class BadClass:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        # ... many attributes
    
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    # ... many methods (god class antipattern)

# Global variables (code smell)
global_counter = 0
global_data = {}
global_config = None
'''
    
    with open(samples_dir / "quality_issues.py", "w") as f:
        f.write(quality_sample)
    
    # Sample 4: Good code (for comparison)
    good_sample = '''
"""
Well-written Python code following best practices
"""

from typing import List, Optional, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class User:
    """Represents a user in the system"""
    id: int
    name: str
    email: str
    
    def validate_email(self) -> bool:
        """Validate user email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return re.match(pattern, self.email) is not None

class UserService:
    """Service for managing users"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.users: Dict[int, User] = {}
    
    def create_user(self, name: str, email: str) -> Optional[User]:
        """Create a new user with validation"""
        if not name or not email:
            logger.warning("Invalid user data provided")
            return None
        
        user = User(
            id=len(self.users) + 1,
            name=name,
            email=email
        )
        
        if user.validate_email():
            self.users[user.id] = user
            logger.info(f"Created user: {user.name}")
            return user
        else:
            logger.error(f"Invalid email format: {email}")
            return None
    
    def find_users_by_name(self, name: str) -> List[User]:
        """Find users by name efficiently"""
        return [
            user for user in self.users.values()
            if name.lower() in user.name.lower()
        ]

def process_data_efficiently(items: List[str]) -> str:
    """Process data using efficient methods"""
    # Use join instead of string concatenation
    return " ".join(items)

def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers"""
    try:
        if b == 0:
            logger.warning("Division by zero attempted")
            return None
        return a / b
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid input for division: {e}")
        return None
'''
    
    with open(samples_dir / "good_code.py", "w") as f:
        f.write(good_sample)
    
    print(f"✅ Created sample code files in {samples_dir}/")
    return samples_dir

def demo_basic_analysis():
    """Demo basic file analysis"""
    print("\\n" + "="*60)
    print("🔍 DEMO 1: BASIC FILE ANALYSIS")
    print("="*60)
    
    # Create sample files
    samples_dir = create_sample_code_files()
    
    # Initialize ML Lint Agent
    print("\\n🤖 Initializing ML Lint Agent...")
    agent = MLLintAgent()
    
    # Analyze security issues file
    print("\\n📁 Analyzing security_issues.py...")
    security_file = samples_dir / "security_issues.py"
    issues = agent.analyze_file(str(security_file), developer_id="default")
    
    print(f"\\n📊 Found {len(issues)} issues:")
    for i, issue in enumerate(issues[:5], 1):  # Show top 5
        print(f"\\n{i}. 🚨 {issue.title}")
        print(f"   📍 Line {issue.line_number}: {issue.severity.upper()}")
        print(f"   📝 {issue.description}")
        print(f"   💡 Fix: {issue.suggested_fix}")
        print(f"   🎯 Confidence: {issue.confidence:.2f}")

def demo_personalized_analysis():
    """Demo personalized analysis for different developer types"""
    print("\\n" + "="*60)
    print("👥 DEMO 2: PERSONALIZED ANALYSIS")
    print("="*60)
    
    agent = MLLintAgent()
    samples_dir = Path("sample_code")
    performance_file = samples_dir / "performance_issues.py"
    
    # Analyze for junior developer
    print("\\n👶 Analysis for Junior Developer:")
    junior_issues = agent.analyze_file(str(performance_file), developer_id="junior_dev")
    print(f"   Found {len(junior_issues)} issues (educational focus)")
    
    # Analyze for senior developer  
    print("\\n🧑‍💼 Analysis for Senior Developer:")
    senior_issues = agent.analyze_file(str(performance_file), developer_id="senior_dev")
    print(f"   Found {len(senior_issues)} issues (performance focus)")
    
    # Compare the differences
    print("\\n📊 Comparison:")
    print(f"   Junior dev sees more educational issues")
    print(f"   Senior dev sees fewer basic style issues")

def demo_feedback_learning():
    """Demo learning from developer feedback"""
    print("\\n" + "="*60)
    print("🧠 DEMO 3: LEARNING FROM FEEDBACK")
    print("="*60)
    
    agent = MLLintAgent()
    samples_dir = Path("sample_code")
    quality_file = samples_dir / "quality_issues.py"
    
    # Analyze file
    print("\\n🔍 Initial analysis...")
    issues = agent.analyze_file(str(quality_file), developer_id="test_dev")
    
    if issues:
        # Simulate developer feedback
        print("\\n📝 Simulating developer feedback...")
        
        # Developer accepts first issue
        agent.provide_feedback(
            issue_id=issues[0].id,
            developer_id="test_dev", 
            action="accepted",
            feedback="Good catch! Fixed this issue.",
            fix_applied="Refactored the function to reduce complexity"
        )
        
        # Developer dismisses second issue if exists
        if len(issues) > 1:
            agent.provide_feedback(
                issue_id=issues[1].id,
                developer_id="test_dev",
                action="dismissed", 
                feedback="This is not relevant for our codebase",
            )
        
        print("✅ Feedback recorded and will influence future analysis")

def demo_project_analysis():
    """Demo full project analysis"""
    print("\\n" + "="*60)
    print("📁 DEMO 4: PROJECT ANALYSIS")
    print("="*60)
    
    agent = MLLintAgent()
    samples_dir = Path("sample_code")
    
    # Analyze entire sample project
    print(f"\\n🔍 Analyzing entire project: {samples_dir}")
    result = agent.analyze_project(str(samples_dir), developer_id="default")
    
    print(f"\\n📊 Project Analysis Results:")
    print(f"   Files analyzed: {result['files_analyzed']}")
    print(f"   Total issues: {len(result['issues'])}")
    
    # Group issues by type
    issue_types = {}
    for issue in result['issues']:
        issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
    
    print("\\n📈 Issues by category:")
    for issue_type, count in sorted(issue_types.items()):
        print(f"   {issue_type}: {count}")
    
    # Show most critical issues
    critical_issues = [i for i in result['issues'] if i.severity == 'critical']
    if critical_issues:
        print(f"\\n🚨 {len(critical_issues)} Critical Issues Found:")
        for issue in critical_issues[:3]:
            print(f"   • {issue.title} (Line {issue.line_number})")

def demo_security_analysis():
    """Demo advanced security analysis"""
    print("\\n" + "="*60)
    print("🔒 DEMO 5: SECURITY ANALYSIS")
    print("="*60)
    
    agent = MLLintAgent()
    samples_dir = Path("sample_code")
    security_file = samples_dir / "security_issues.py"
    
    print("\\n🔍 Running ML-powered security analysis...")
    issues = agent.analyze_file(str(security_file), developer_id="security_expert")
    
    # Filter security issues
    security_issues = [i for i in issues if 'security' in i.issue_type.lower() or 'security' in ' '.join(i.tags)]
    
    print(f"\\n🛡️  Found {len(security_issues)} security-related issues:")
    for issue in security_issues:
        print(f"\\n🚨 {issue.title}")
        print(f"   Severity: {issue.severity.upper()}")
        print(f"   Description: {issue.description}")
        print(f"   Line: {issue.line_number}")
        print(f"   Fix: {issue.suggested_fix}")
        if issue.references:
            print(f"   References: {', '.join(issue.references)}")

def demo_performance_analysis():
    """Demo performance optimization suggestions"""
    print("\\n" + "="*60)
    print("⚡ DEMO 6: PERFORMANCE ANALYSIS")
    print("="*60)
    
    agent = MLLintAgent()
    samples_dir = Path("sample_code")
    performance_file = samples_dir / "performance_issues.py"
    
    print("\\n🔍 Running ML-powered performance analysis...")
    issues = agent.analyze_file(str(performance_file), developer_id="performance_expert")
    
    # Filter performance issues
    perf_issues = [i for i in issues if 'performance' in i.issue_type.lower() or 'performance' in ' '.join(i.tags)]
    
    print(f"\\n⚡ Found {len(perf_issues)} performance-related suggestions:")
    for issue in perf_issues:
        print(f"\\n🎯 {issue.title}")
        print(f"   Impact: {issue.severity}")
        print(f"   Description: {issue.description}")
        print(f"   Optimization: {issue.suggested_fix}")

def demo_comparison_with_traditional_linters():
    """Demo comparison with traditional linting"""
    print("\\n" + "="*60)
    print("⚖️  DEMO 7: ML LINT vs TRADITIONAL LINT")
    print("="*60)
    
    print("\\n🤖 ML Lint Agent Features:")
    print("   ✅ Context-aware analysis")
    print("   ✅ Learning from feedback")
    print("   ✅ Natural language explanations")
    print("   ✅ Personalized suggestions")
    print("   ✅ Security pattern recognition")
    print("   ✅ Performance optimization hints")
    print("   ✅ Code quality insights")
    print("   ✅ Project-wide analysis")
    print("   ✅ Confidence scoring")
    print("   ✅ Smart prioritization")
    
    print("\\n📐 Traditional Linters:")
    print("   ⚠️  Rule-based only")
    print("   ⚠️  No learning capability")
    print("   ⚠️  Generic error messages")
    print("   ⚠️  One-size-fits-all")
    print("   ⚠️  Limited security detection")
    print("   ⚠️  Basic performance checks")
    print("   ⚠️  Style-focused")
    print("   ⚠️  File-by-file analysis")
    print("   ⚠️  No confidence measurement")
    print("   ⚠️  Equal priority for all issues")

def interactive_demo():
    """Interactive demo allowing user to test the agent"""
    print("\\n" + "="*60)
    print("🎮 INTERACTIVE ML LINT AGENT DEMO")
    print("="*60)
    
    agent = MLLintAgent()
    
    print("\\nType Python code and get ML-powered analysis!")
    print("Type 'quit' to exit, 'demo' for sample code")
    
    while True:
        try:
            print("\\n" + "-"*50)
            user_input = input("🐍 Enter Python code (or 'quit'/'demo'): ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thanks for trying the ML Lint Agent!")
                break
            
            if user_input.lower() == 'demo':
                # Show demo code with issues
                demo_code = '''def bad_function(a, b, c, d, e):
    password = "secret123"
    if a and b and c and d and e:
        try:
            result = eval(user_input)
            for i in range(len(items)):
                print(items[i])
        except:
            pass
    return result'''
                print("\\n📝 Demo code with issues:")
                print(demo_code)
                user_input = demo_code
            
            if not user_input:
                continue
            
            # Write code to temporary file
            temp_file = "temp_analysis.py"
            with open(temp_file, 'w') as f:
                f.write(user_input)
            
            # Analyze
            print("\\n🤖 Analyzing code with ML...")
            issues = agent.analyze_file(temp_file, developer_id="interactive_user")
            
            if issues:
                print(f"\\n📊 Found {len(issues)} issues:")
                for i, issue in enumerate(issues, 1):
                    print(f"\\n{i}. 🚨 {issue.title}")
                    print(f"   Severity: {issue.severity.upper()}")
                    print(f"   Line: {issue.line_number}")
                    print(f"   Issue: {issue.description}")
                    print(f"   Fix: {issue.suggested_fix}")
                    print(f"   Confidence: {issue.confidence:.2f}")
            else:
                print("\\n✅ No issues found! Clean code! 🎉")
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except KeyboardInterrupt:
            print("\\n👋 Thanks for trying the ML Lint Agent!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    print("🚀 ML-INTEGRATED LINT AGENT DEMONSTRATION")
    print("="*60)
    print("This demo showcases an advanced lint agent powered by your fine-tuned CodeT5+ model")
    print("\\nFeatures demonstrated:")
    print("• Intelligent code review with pattern recognition")
    print("• Context-aware analysis across files") 
    print("• Learning from developer feedback")
    print("• Advanced refactoring suggestions")
    print("• Security & bug detection")
    print("• Natural language explanations")
    print("• Smart prioritization")
    print("• Personalized feedback")
    
    demos = [
        ("1", "Basic File Analysis", demo_basic_analysis),
        ("2", "Personalized Analysis", demo_personalized_analysis), 
        ("3", "Learning from Feedback", demo_feedback_learning),
        ("4", "Project Analysis", demo_project_analysis),
        ("5", "Security Analysis", demo_security_analysis),
        ("6", "Performance Analysis", demo_performance_analysis),
        ("7", "ML vs Traditional Comparison", demo_comparison_with_traditional_linters),
        ("8", "Interactive Demo", interactive_demo),
        ("9", "Run All Demos", None)
    ]
    
    print("\\n📋 Available Demos:")
    for num, title, _ in demos:
        print(f"   {num}. {title}")
    
    while True:
        try:
            choice = input("\\n🎯 Choose demo (1-9) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Thanks for exploring the ML Lint Agent!")
                break
            
            if choice == "9":
                # Run all demos
                for num, title, demo_func in demos[:-2]:  # Exclude interactive and "all"
                    if demo_func:
                        demo_func()
                break
            
            # Find and run selected demo
            demo_found = False
            for num, title, demo_func in demos:
                if choice == num and demo_func:
                    demo_func()
                    demo_found = True
                    break
            
            if not demo_found:
                print("❌ Invalid choice. Please select 1-9 or 'q'")
                
        except KeyboardInterrupt:
            print("\\n👋 Thanks for exploring the ML Lint Agent!")
            break
        except Exception as e:
            print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main()
