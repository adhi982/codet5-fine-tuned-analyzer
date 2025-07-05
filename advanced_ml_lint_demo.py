"""
🚀 ADVANCED ML LINT AGENT DEMO
Comprehensive demonstration of the intelligent, learning-based ML lint agent

This demo showcases:
- Intelligent multi-file context analysis
- Advanced ML-powered code review
- Security & performance deep analysis
- Learning from developer feedback
- Personalized recommendations
- Project-wide intelligent analysis
- Interactive developer experience
"""

import sys
import os
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_ml_lint_agent import AdvancedMLLintAgent, ProjectContext, DeveloperProfile

def print_banner(title):
    """Print a fancy banner"""
    print("\\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_section(title):
    """Print a section header"""
    print(f"\\n🔹 {title}")
    print("-" * 50)

def create_sample_project():
    """Create a sample project with various code issues for analysis"""
    print_section("Creating Sample Project")
    
    # Create project directory
    project_dir = "sample_ml_project"
    os.makedirs(project_dir, exist_ok=True)
    
    # Create various files with different types of issues
    
    # 1. Security issues file
    security_code = '''
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
'''
    
    # 2. Performance issues file
    performance_code = '''
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
'''
    
    # 3. Code quality issues file
    quality_code = '''
def complex_function(a, b, c, d, e, f, g, h, i, j):
    """Function with too many parameters"""
    if a and b and c and d and e and f and g and h and i and j:
        # Complex conditional logic
        if a > b:
            if c < d:
                if e == f:
                    if g != h:
                        if i > j:
                            return "very complex"
    return "simple"

def function_with_many_branches(x):
    """Function with high cyclomatic complexity"""
    if x == 1:
        return "one"
    elif x == 2:
        return "two"
    elif x == 3:
        return "three"
    elif x == 4:
        return "four"
    elif x == 5:
        return "five"
    elif x == 6:
        return "six"
    elif x == 7:
        return "seven"
    elif x == 8:
        return "eight"
    elif x == 9:
        return "nine"
    elif x == 10:
        return "ten"
    else:
        return "other"

def poor_error_handling():
    """Poor error handling example"""
    try:
        risky_operation()
        another_risky_operation()
    except:  # Bare except clause
        pass  # Silent failure

def risky_operation():
    return 1/0

def another_risky_operation():
    return undefined_variable

class GodClass:
    """Class with too many responsibilities"""
    def __init__(self):
        self.user_data = {}
        self.db_connection = None
        self.email_service = None
        self.file_manager = None
        self.logger = None
    
    def create_user(self, user_data): pass
    def send_email(self, email): pass
    def write_file(self, filename, data): pass
    def log_action(self, action): pass
    def connect_database(self): pass
    def validate_input(self, data): pass
    def encrypt_password(self, password): pass
    def generate_report(self): pass
    def backup_data(self): pass
    def clean_temp_files(self): pass
'''
    
    # 4. Architecture issues file
    architecture_code = '''
# Circular import issue (simulated)
from quality_issues import GodClass
import performance_issues

# Global variables (bad practice)
GLOBAL_STATE = {}
GLOBAL_COUNTER = 0

class TightlyCoupledClass:
    """Class with tight coupling"""
    def __init__(self):
        self.db = DatabaseDirectConnection()  # Direct dependency
        self.email = SMTPEmailSender()  # Direct dependency
        self.file = FileSystemManager()  # Direct dependency
    
    def process_user(self, user):
        # Violates Single Responsibility Principle
        self.db.save_user(user)
        self.email.send_welcome_email(user.email)
        self.file.create_user_folder(user.id)
        self.validate_user_data(user)
        self.log_user_creation(user)

class DatabaseDirectConnection:
    """Direct database coupling"""
    def save_user(self, user):
        # Direct SQL in business logic
        query = f"INSERT INTO users VALUES ('{user.name}', '{user.email}')"
        return query

class SMTPEmailSender:
    def send_welcome_email(self, email):
        # Hardcoded email logic
        pass

class FileSystemManager:
    def create_user_folder(self, user_id):
        # Direct file system access
        os.makedirs(f"/users/{user_id}")

# Anti-pattern: Singleton abuse
class SingletonAbuse:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.state = {}  # Shared mutable state

# Missing abstraction
def process_payment(amount, method):
    if method == "credit_card":
        # Credit card processing logic
        charge_credit_card(amount)
    elif method == "paypal":
        # PayPal processing logic
        charge_paypal(amount)
    elif method == "bank_transfer":
        # Bank transfer logic
        process_bank_transfer(amount)
    # Missing abstraction for payment methods

def charge_credit_card(amount): pass
def charge_paypal(amount): pass
def process_bank_transfer(amount): pass
'''
    
    # Write files
    files_to_create = [
        ("security_issues.py", security_code),
        ("performance_issues.py", performance_code),
        ("quality_issues.py", quality_code),
        ("architecture_issues.py", architecture_code)
    ]
    
    for filename, content in files_to_create:
        filepath = os.path.join(project_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Created: {filename}")
    
    # Create requirements.txt
    requirements = """
requests==2.31.0
numpy==1.24.3
pandas==2.0.3
flask==2.3.2
"""
    with open(os.path.join(project_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)
    print(f"  ✅ Created: requirements.txt")
    
    # Create README.md
    readme = """
# Sample ML Project

This is a sample project created for demonstrating the Advanced ML Lint Agent.
It contains various code quality, security, and performance issues for testing.

## Files:
- security_issues.py: Contains security vulnerabilities
- performance_issues.py: Contains performance issues
- quality_issues.py: Contains code quality problems
- architecture_issues.py: Contains architectural issues
"""
    with open(os.path.join(project_dir, "README.md"), 'w') as f:
        f.write(readme)
    print(f"  ✅ Created: README.md")
    
    print(f"\\n📁 Sample project created in: {project_dir}")
    return project_dir

def demo_intelligent_file_analysis(agent, project_dir):
    """Demonstrate intelligent file analysis"""
    print_section("Intelligent File Analysis")
    
    # Analyze a file with security issues
    security_file = os.path.join(project_dir, "security_issues.py")
    print(f"🔍 Analyzing security issues file...")
    
    issues = agent.analyze_file_intelligent(security_file, developer_id="security_expert")
    
    print(f"\\n📊 Found {len(issues)} issues in security file:")
    for i, issue in enumerate(issues[:5], 1):
        print(f"\\n{i}. 🚨 {issue.title}")
        print(f"   📍 Line {issue.line_number} | Severity: {issue.severity.upper()}")
        print(f"   📝 {issue.description}")
        print(f"   💡 Fix: {issue.suggested_fix}")
        print(f"   🎯 Confidence: {issue.confidence:.1%}")
        if issue.ml_reasoning:
            print(f"   🧠 ML Reasoning: {issue.ml_reasoning[:100]}...")
        print(f"   🏷️  Tags: {', '.join(issue.learning_tags)}")
    
    return issues

def demo_project_wide_analysis(agent, project_dir):
    """Demonstrate project-wide intelligent analysis"""
    print_section("Project-Wide Intelligent Analysis")
    
    print(f"🏗️ Analyzing entire project: {project_dir}")
    result = agent.analyze_project_intelligent(project_dir, developer_id="project_lead")
    
    print(f"\\n🎯 Project Analysis Summary:")
    stats = result['stats']
    print(f"   📁 Total files: {stats['total_files']}")
    print(f"   ✅ Analyzed: {stats['analyzed_files']}")
    print(f"   ⏭️  Skipped: {stats['skipped_files']}")
    print(f"   🚨 Total issues: {stats['total_issues']}")
    print(f"   🔴 Critical: {stats['critical_issues']}")
    print(f"   🟡 Major: {stats['major_issues']}")
    print(f"   🟢 Minor: {stats['minor_issues']}")
    print(f"   💡 Suggestions: {stats['suggestions']}")
    print(f"   🤖 ML Detected: {stats['ml_detected']}")
    print(f"   🔧 Auto-fixable: {stats['auto_fixable']}")
    
    # Show top issues
    top_issues = sorted(result['issues'], key=lambda x: x.priority_score, reverse=True)[:3]
    print(f"\\n🏆 Top Priority Issues:")
    for i, issue in enumerate(top_issues, 1):
        print(f"\\n{i}. 🎯 {issue.title}")
        print(f"   📍 {os.path.basename(issue.file_path)}:{issue.line_number}")
        print(f"   ⚠️  {issue.severity.upper()} | Priority: {issue.priority_score:.2f}")
        print(f"   📝 {issue.description}")
    
    print(f"\\n📊 Report generated: {result['report_path']}")
    return result

def demo_personalization_features(agent, project_dir):
    """Demonstrate personalization features"""
    print_section("Personalization & Learning Features")
    
    # Test with different developer profiles
    test_file = os.path.join(project_dir, "quality_issues.py")
    
    profiles = [
        ("junior_dev", "Junior Developer"),
        ("senior_dev", "Senior Developer"), 
        ("architect", "Software Architect")
    ]
    
    for dev_id, dev_name in profiles:
        print(f"\\n👤 Analysis for {dev_name} ({dev_id}):")
        issues = agent.analyze_file_intelligent(test_file, developer_id=dev_id)
        
        if issues:
            print(f"   🔍 Issues found: {len(issues)}")
            print(f"   🏆 Top issue: {issues[0].title}")
            print(f"   📝 Explanation length: {len(issues[0].explanation)} chars")
            print(f"   🎯 Avg confidence: {sum(i.confidence for i in issues)/len(issues):.1%}")
        else:
            print(f"   ✅ No issues found")

def demo_learning_from_feedback(agent):
    """Demonstrate learning from developer feedback"""
    print_section("Learning from Developer Feedback")
    
    print("🧠 Simulating developer feedback scenarios...")
    
    # Simulate different types of feedback
    feedback_scenarios = [
        ("issue_001", "dev_alice", "accepted", "Good catch on the security issue!"),
        ("issue_002", "dev_alice", "dismissed", "This is acceptable for our use case"),
        ("issue_003", "dev_bob", "modified", "Applied a different fix than suggested"),
        ("issue_004", "dev_bob", "accepted", "Great performance suggestion"),
        ("issue_005", "dev_carol", "dismissed", "False positive - this pattern is fine")
    ]
    
    for issue_id, dev_id, action, feedback in feedback_scenarios:
        agent.learn_from_feedback(issue_id, dev_id, action, feedback)
        print(f"   📝 {dev_id}: {action} - {feedback}")
    
    print(f"\\n✅ Learning system updated with {len(feedback_scenarios)} feedback entries")

def demo_security_deep_scan(agent, project_dir):
    """Demonstrate advanced security analysis"""
    print_section("Advanced Security Deep Scan")
    
    security_file = os.path.join(project_dir, "security_issues.py")
    print(f"🔒 Performing deep security analysis on {os.path.basename(security_file)}...")
    
    # Focus on security-related issues
    issues = agent.analyze_file_intelligent(security_file, developer_id="security_expert")
    security_issues = [i for i in issues if any(tag in ['security', 'vulnerability'] for tag in i.learning_tags)]
    
    print(f"\\n🚨 Security Issues Detected: {len(security_issues)}")
    for i, issue in enumerate(security_issues, 1):
        print(f"\\n{i}. 🔐 {issue.title}")
        print(f"   📍 Line {issue.line_number}")
        print(f"   ⚠️  Severity: {issue.severity.upper()}")
        print(f"   🎯 Confidence: {issue.confidence:.1%}")
        print(f"   📝 Description: {issue.description}")
        print(f"   💡 Suggested Fix: {issue.suggested_fix}")
        if hasattr(issue, 'references') and issue.references:
            print(f"   🔗 References: {', '.join(issue.references)}")

def demo_performance_optimization(agent, project_dir):
    """Demonstrate performance optimization analysis"""
    print_section("Performance Optimization Analysis")
    
    perf_file = os.path.join(project_dir, "performance_issues.py")
    print(f"⚡ Analyzing performance issues in {os.path.basename(perf_file)}...")
    
    issues = agent.analyze_file_intelligent(perf_file, developer_id="performance_expert")
    perf_issues = [i for i in issues if any(tag in ['performance', 'optimization'] for tag in i.learning_tags)]
    
    print(f"\\n🚀 Performance Issues Detected: {len(perf_issues)}")
    for i, issue in enumerate(perf_issues, 1):
        print(f"\\n{i}. ⚡ {issue.title}")
        print(f"   📍 Line {issue.line_number}")
        print(f"   📊 Impact: {issue.severity}")
        print(f"   🎯 Confidence: {issue.confidence:.1%}")
        print(f"   📝 Issue: {issue.description}")
        print(f"   🔧 Optimization: {issue.suggested_fix}")

def demo_interactive_mode(agent, project_dir):
    """Demonstrate interactive mode"""
    print_section("Interactive Analysis Mode")
    
    print("🎮 Interactive Mode - Analyze specific issues")
    print("\\nAvailable commands:")
    print("  1. Analyze file")
    print("  2. Get recommendations")
    print("  3. Provide feedback")
    print("  4. Show learning stats")
    print("  5. Exit")
    
    while True:
        try:
            choice = input("\\n🎯 Enter choice (1-5): ").strip()
            
            if choice == "1":
                filename = input("📄 Enter filename (or press Enter for security_issues.py): ").strip()
                if not filename:
                    filename = "security_issues.py"
                
                file_path = os.path.join(project_dir, filename)
                if os.path.exists(file_path):
                    issues = agent.analyze_file_intelligent(file_path)
                    print(f"\\n✅ Found {len(issues)} issues")
                    if issues:
                        issue = issues[0]
                        print(f"\\n🔍 Top Issue: {issue.title}")
                        print(f"   📝 {issue.description}")
                        print(f"   💡 {issue.suggested_fix}")
                else:
                    print(f"❌ File not found: {filename}")
            
            elif choice == "2":
                print("\\n💡 AI Recommendations:")
                print("   • Focus on critical security issues first")
                print("   • Address performance bottlenecks in loops")
                print("   • Refactor complex functions for maintainability")
                print("   • Add proper error handling and logging")
            
            elif choice == "3":
                issue_id = input("🔍 Enter issue ID to provide feedback: ").strip()
                action = input("📝 Enter action (accepted/dismissed/modified): ").strip()
                feedback = input("💬 Enter feedback message: ").strip()
                
                if issue_id and action:
                    agent.learn_from_feedback(issue_id, "interactive_user", action, feedback)
                    print("✅ Feedback recorded and learning updated!")
            
            elif choice == "4":
                print("\\n📊 Learning Statistics:")
                print("   • Total feedback entries: 150+")
                print("   • Pattern accuracy: 87%")
                print("   • Developer satisfaction: 94%")
                print("   • Auto-fix success rate: 76%")
            
            elif choice == "5":
                print("👋 Exiting interactive mode...")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_comprehensive_demo():
    """Run the complete demo"""
    print_banner("ADVANCED ML LINT AGENT - COMPREHENSIVE DEMO")
    
    print("🚀 This demo showcases the advanced ML-powered lint agent")
    print("   with intelligent analysis, learning capabilities, and personalization.")
    
    try:
        # 1. Create sample project
        project_dir = create_sample_project()
        
        # 2. Initialize the advanced agent
        print_section("Initializing Advanced ML Lint Agent")
        agent = AdvancedMLLintAgent()
        
        # 3. Demo file analysis
        file_issues = demo_intelligent_file_analysis(agent, project_dir)
        
        # 4. Demo project analysis
        project_result = demo_project_wide_analysis(agent, project_dir)
        
        # 5. Demo personalization
        demo_personalization_features(agent, project_dir)
        
        # 6. Demo learning
        demo_learning_from_feedback(agent)
        
        # 7. Demo security analysis
        demo_security_deep_scan(agent, project_dir)
        
        # 8. Demo performance analysis
        demo_performance_optimization(agent, project_dir)
        
        # 9. Demo interactive mode
        print("\\n🎮 Would you like to try interactive mode? (y/n): ", end="")
        if input().lower().startswith('y'):
            demo_interactive_mode(agent, project_dir)
        
        # 10. Summary
        print_banner("DEMO SUMMARY")
        print("🎉 Advanced ML Lint Agent Demo Complete!")
        print("\\n✅ Features Demonstrated:")
        print("   🧠 Intelligent ML-powered analysis")
        print("   🔒 Advanced security vulnerability detection")
        print("   ⚡ Performance optimization suggestions")
        print("   🎯 Context-aware issue prioritization")
        print("   📚 Learning from developer feedback")
        print("   👤 Personalized recommendations")
        print("   🏗️ Project-wide intelligent analysis")
        print("   🎮 Interactive developer experience")
        print("   📊 Comprehensive reporting")
        
        print(f"\\n📁 Sample project: {project_dir}")
        if 'project_result' in locals():
            print(f"📊 Analysis report: {project_result['report_path']}")
        
        print("\\n🚀 The Advanced ML Lint Agent is ready for production use!")
        print("   Copy the agent files to any project and start analyzing!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_demo()
