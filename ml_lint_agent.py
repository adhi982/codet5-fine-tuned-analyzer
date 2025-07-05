"""
ML-INTEGRATED LINT AGENT
Advanced code analysis using your fine-tuned CodeT5+ model

Features:
- Intelligent code review with pattern recognition
- Context-aware analysis across files
- Learning from feedback history
- Advanced refactoring suggestions
- Security & bug detection
- Natural language explanations
- Smart prioritization
- Personalized feedback
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
import ast
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class CodeIssue:
    """Represents a code issue found by the ML agent"""
    id: str
    file_path: str
    line_number: int
    column: int
    issue_type: str
    severity: str  # 'critical', 'major', 'minor', 'suggestion'
    title: str
    description: str
    explanation: str
    suggested_fix: str
    confidence: float
    code_snippet: str
    context: str
    references: List[str]
    tags: List[str]
    timestamp: str

@dataclass
class DeveloperFeedback:
    """Stores developer feedback on issues"""
    issue_id: str
    developer_id: str
    action: str  # 'accepted', 'dismissed', 'modified'
    feedback: str
    timestamp: str
    fix_applied: Optional[str] = None

class MLLintAgent:
    """
    Advanced ML-powered lint agent using fine-tuned CodeT5+
    """
    
    def __init__(self, model_path=None, config_path="./ml_lint_config.json"):
        """Initialize the ML Lint Agent"""
        print("🤖 Initializing ML-Integrated Lint Agent...")
        
        # Load the fine-tuned model
        self.model_path = model_path or self._find_model_path()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize components
        self.config_path = config_path
        self.config = self._load_config()
        self.feedback_history = self._load_feedback_history()
        self.knowledge_base = self._build_knowledge_base()
        self.pattern_database = self._load_patterns()
        
        print(f"✅ ML Lint Agent loaded on {self.device}")
        print(f"📊 Feedback history: {len(self.feedback_history)} entries")
        print(f"🧠 Knowledge base: {len(self.knowledge_base)} patterns")
    
    def _find_model_path(self):
        """Auto-find the model path"""
        possible_paths = [
            "./models/codet5p-finetuned",
            "./codet5p-finetuned",
            "../models/codet5p-finetuned",
            "E:/Intel Fest/Fine Tune/checkpoints/codet5p-finetuned"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Fine-tuned model not found!")
    
    def _load_config(self):
        """Load agent configuration"""
        default_config = {
            "severity_weights": {"critical": 1.0, "major": 0.8, "minor": 0.5, "suggestion": 0.3},
            "confidence_threshold": 0.6,
            "max_issues_per_file": 20,
            "learning_rate": 0.1,
            "developer_profiles": {},
            "project_conventions": {},
            "security_patterns": [],
            "performance_patterns": [],
            "style_patterns": []
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        
        return default_config
    
    def _load_feedback_history(self):
        """Load historical feedback for learning"""
        feedback_path = "ml_lint_feedback.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, 'r') as f:
                return json.load(f)
        return []
    
    def _build_knowledge_base(self):
        """Build knowledge base of code patterns and issues"""
        return {
            "security_vulnerabilities": [
                {"pattern": r"eval\\(.*\\)", "description": "Use of eval() can lead to code injection", "severity": "critical"},
                {"pattern": r"subprocess\\.call\\(.*shell=True.*\\)", "description": "Shell injection vulnerability", "severity": "critical"},
                {"pattern": r"pickle\\.loads?\\(.*\\)", "description": "Unsafe pickle deserialization", "severity": "major"},
                {"pattern": r"password.*=.*[\"'].*[\"']", "description": "Hardcoded password detected", "severity": "major"},
            ],
            "performance_issues": [
                {"pattern": r"for.*in.*range\\(len\\(.*\\)\\)", "description": "Use enumerate() instead of range(len())", "severity": "minor"},
                {"pattern": r"\\.append\\(.*\\) in.*for.*loop", "description": "Consider list comprehension for better performance", "severity": "suggestion"},
                {"pattern": r"\\+.*str.*in.*loop", "description": "Use join() instead of string concatenation in loops", "severity": "minor"},
            ],
            "code_smells": [
                {"pattern": r"def.*\\(.*,.*,.*,.*,.*,.*\\)", "description": "Too many parameters, consider refactoring", "severity": "minor"},
                {"pattern": r"if.*and.*and.*and.*and", "description": "Complex condition, consider extracting to variable", "severity": "suggestion"},
                {"pattern": r"try:.*\\n.*except:.*\\n.*pass", "description": "Empty except clause hides errors", "severity": "major"},
            ]
        }
    
    def _load_patterns(self):
        """Load learned patterns from feedback"""
        patterns_path = "learned_patterns.pkl"
        if os.path.exists(patterns_path):
            with open(patterns_path, 'rb') as f:
                return pickle.load(f)
        return {"good_patterns": [], "bad_patterns": []}
    
    def analyze_file(self, file_path: str, developer_id: str = "default") -> List[CodeIssue]:
        """
        Analyze a single file and return issues
        
        Args:
            file_path: Path to the file to analyze
            developer_id: ID of the developer (for personalization)
            
        Returns:
            List of CodeIssue objects
        """
        print(f"🔍 Analyzing: {file_path}")
        
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        issues = []
        
        # 1. Static pattern matching
        pattern_issues = self._find_pattern_issues(file_path, code_content)
        issues.extend(pattern_issues)
        
        # 2. ML-based analysis
        ml_issues = self._ml_analyze_code(file_path, code_content)
        issues.extend(ml_issues)
        
        # 3. Context-aware analysis
        context_issues = self._analyze_context(file_path, code_content)
        issues.extend(context_issues)
        
        # 4. Security analysis
        security_issues = self._analyze_security(file_path, code_content)
        issues.extend(security_issues)
        
        # 5. Performance analysis
        performance_issues = self._analyze_performance(file_path, code_content)
        issues.extend(performance_issues)
        
        # 6. Apply learning and personalization
        personalized_issues = self._personalize_issues(issues, developer_id)
        
        # 7. Rank and filter issues
        final_issues = self._rank_and_filter_issues(personalized_issues)
        
        print(f"✅ Found {len(final_issues)} issues in {file_path}")
        return final_issues
    
    def _find_pattern_issues(self, file_path: str, code: str) -> List[CodeIssue]:
        """Find issues using pattern matching"""
        issues = []
        lines = code.split('\\n')
        
        for category, patterns in self.knowledge_base.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                description = pattern_info["description"]
                severity = pattern_info["severity"]
                
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        issue = CodeIssue(
                            id=self._generate_issue_id(file_path, line_num, pattern),
                            file_path=file_path,
                            line_number=line_num,
                            column=0,
                            issue_type=category,
                            severity=severity,
                            title=f"{category.replace('_', ' ').title()} detected",
                            description=description,
                            explanation=self._generate_explanation(pattern, line, category),
                            suggested_fix=self._generate_fix_suggestion(pattern, line),
                            confidence=0.9,
                            code_snippet=line.strip(),
                            context=self._get_context(lines, line_num),
                            references=[],
                            tags=[category, severity],
                            timestamp=datetime.now().isoformat()
                        )
                        issues.append(issue)
        
        return issues
    
    def _ml_analyze_code(self, file_path: str, code: str) -> List[CodeIssue]:
        """Use ML model to analyze code quality"""
        issues = []
        
        # Split code into functions and classes for analysis
        code_blocks = self._extract_code_blocks(code)
        
        for block in code_blocks:
            # Use the fine-tuned model to analyze the code block
            analysis_prompt = f"Analyze this code for issues:\\n{block['code']}"
            
            try:
                inputs = self.tokenizer(analysis_prompt, return_tensors="pt", max_length=400, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.3,  # Lower temperature for more focused analysis
                        do_sample=True,
                        repetition_penalty=1.2
                    )
                
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                analysis = result.split("Analyze this code for issues:")[-1].strip()
                
                # Parse the ML output to extract issues
                ml_issues = self._parse_ml_analysis(file_path, block, analysis)
                issues.extend(ml_issues)
                
            except Exception as e:
                print(f"⚠️  ML analysis failed for block: {e}")
        
        return issues
    
    def _analyze_context(self, file_path: str, code: str) -> List[CodeIssue]:
        """Analyze code in context of the entire project"""
        issues = []
        
        try:
            # Parse the AST to understand code structure
            tree = ast.parse(code)
            
            # Analyze imports
            imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
            import_issues = self._analyze_imports(file_path, imports)
            issues.extend(import_issues)
            
            # Analyze function complexity
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            complexity_issues = self._analyze_complexity(file_path, functions, code)
            issues.extend(complexity_issues)
            
            # Analyze variable usage
            variables = self._extract_variables(tree)
            variable_issues = self._analyze_variables(file_path, variables, code)
            issues.extend(variable_issues)
            
        except SyntaxError:
            # If code has syntax errors, create an issue for that
            issue = CodeIssue(
                id=self._generate_issue_id(file_path, 1, "syntax_error"),
                file_path=file_path,
                line_number=1,
                column=0,
                issue_type="syntax_error",
                severity="critical",
                title="Syntax Error",
                description="File contains syntax errors",
                explanation="The code contains syntax errors that prevent proper parsing",
                suggested_fix="Fix syntax errors first",
                confidence=1.0,
                code_snippet="",
                context="",
                references=[],
                tags=["syntax", "critical"],
                timestamp=datetime.now().isoformat()
            )
            issues.append(issue)
        
        return issues
    
    def _analyze_security(self, file_path: str, code: str) -> List[CodeIssue]:
        """Advanced security analysis"""
        issues = []
        
        # Use ML model for security analysis
        security_prompt = f"Find security vulnerabilities in this code:\\n{code[:1000]}"
        
        try:
            inputs = self.tokenizer(security_prompt, return_tensors="pt", max_length=400, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.2,
                    do_sample=True
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            security_analysis = result.split("Find security vulnerabilities in this code:")[-1].strip()
            
            # Parse security findings
            if "vulnerability" in security_analysis.lower() or "security" in security_analysis.lower():
                issue = CodeIssue(
                    id=self._generate_issue_id(file_path, 1, "security_ml"),
                    file_path=file_path,
                    line_number=1,
                    column=0,
                    issue_type="security",
                    severity="major",
                    title="Potential Security Issue",
                    description="ML model detected potential security concerns",
                    explanation=f"Security analysis: {security_analysis}",
                    suggested_fix="Review code for security best practices",
                    confidence=0.7,
                    code_snippet=code[:200] + "...",
                    context="Full file analysis",
                    references=["https://owasp.org/", "https://security.python.org/"],
                    tags=["security", "ml-detected"],
                    timestamp=datetime.now().isoformat()
                )
                issues.append(issue)
        
        except Exception as e:
            print(f"⚠️  Security analysis failed: {e}")
        
        return issues
    
    def _analyze_performance(self, file_path: str, code: str) -> List[CodeIssue]:
        """Performance analysis using ML"""
        issues = []
        
        # Use ML model for performance analysis
        perf_prompt = f"Suggest performance improvements for this code:\\n{code[:800]}"
        
        try:
            inputs = self.tokenizer(perf_prompt, return_tensors="pt", max_length=400, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120,
                    temperature=0.4,
                    do_sample=True
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            perf_analysis = result.split("Suggest performance improvements for this code:")[-1].strip()
            
            # Parse performance suggestions
            if any(word in perf_analysis.lower() for word in ["optimize", "improve", "faster", "efficient"]):
                issue = CodeIssue(
                    id=self._generate_issue_id(file_path, 1, "performance_ml"),
                    file_path=file_path,
                    line_number=1,
                    column=0,
                    issue_type="performance",
                    severity="suggestion",
                    title="Performance Optimization Opportunity",
                    description="ML model suggests performance improvements",
                    explanation=f"Performance analysis: {perf_analysis}",
                    suggested_fix=self._generate_performance_fix(perf_analysis),
                    confidence=0.6,
                    code_snippet=code[:200] + "...",
                    context="Performance optimization",
                    references=["https://wiki.python.org/moin/PythonSpeed/PerformanceTips"],
                    tags=["performance", "optimization", "ml-suggested"],
                    timestamp=datetime.now().isoformat()
                )
                issues.append(issue)
        
        except Exception as e:
            print(f"⚠️  Performance analysis failed: {e}")
        
        return issues
    
    def _personalize_issues(self, issues: List[CodeIssue], developer_id: str) -> List[CodeIssue]:
        """Personalize issues based on developer profile and history"""
        developer_profile = self.config.get("developer_profiles", {}).get(developer_id, {})
        experience_level = developer_profile.get("experience_level", "intermediate")
        
        personalized_issues = []
        
        for issue in issues:
            # Adjust severity based on developer experience
            if experience_level == "beginner":
                # Show more educational issues
                if issue.issue_type in ["code_smells", "style"]:
                    issue.severity = "minor"  # Promote educational issues
            elif experience_level == "expert":
                # Filter out basic issues
                if issue.issue_type == "style" and issue.severity == "suggestion":
                    continue  # Skip basic style suggestions for experts
            
            # Learn from past feedback
            similar_feedback = self._find_similar_feedback(issue, developer_id)
            if similar_feedback:
                # Adjust confidence based on past acceptance
                acceptance_rate = self._calculate_acceptance_rate(similar_feedback)
                issue.confidence *= acceptance_rate
            
            # Add personalized explanation
            issue.explanation = self._personalize_explanation(issue, developer_profile)
            
            personalized_issues.append(issue)
        
        return personalized_issues
    
    def _rank_and_filter_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Rank issues by importance and filter low-confidence ones"""
        # Filter by confidence threshold
        filtered_issues = [
            issue for issue in issues 
            if issue.confidence >= self.config["confidence_threshold"]
        ]
        
        # Calculate priority scores
        for issue in filtered_issues:
            severity_weight = self.config["severity_weights"].get(issue.severity, 0.5)
            priority_score = issue.confidence * severity_weight
            issue.priority_score = priority_score
        
        # Sort by priority
        ranked_issues = sorted(filtered_issues, key=lambda x: x.priority_score, reverse=True)
        
        # Limit number of issues per file
        max_issues = self.config["max_issues_per_file"]
        return ranked_issues[:max_issues]
    
    def provide_feedback(self, issue_id: str, developer_id: str, action: str, feedback: str = "", fix_applied: str = None):
        """Record developer feedback for learning"""
        feedback_entry = DeveloperFeedback(
            issue_id=issue_id,
            developer_id=developer_id,
            action=action,
            feedback=feedback,
            timestamp=datetime.now().isoformat(),
            fix_applied=fix_applied
        )
        
        self.feedback_history.append(asdict(feedback_entry))
        self._save_feedback_history()
        
        # Update learning
        self._update_learning_from_feedback(feedback_entry)
        
        print(f"📝 Feedback recorded for issue {issue_id}: {action}")
    
    def generate_report(self, issues: List[CodeIssue], output_path: str = "ml_lint_report.html"):
        """Generate a comprehensive HTML report"""
        html_content = self._generate_html_report(issues)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Report generated: {output_path}")
    
    def analyze_project(self, project_path: str, developer_id: str = "default") -> Dict:
        """Analyze an entire project"""
        print(f"🔍 Analyzing project: {project_path}")
        
        all_issues = []
        file_count = 0
        
        # Find all Python files
        for root, dirs, files in os.walk(project_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_issues = self.analyze_file(file_path, developer_id)
                    all_issues.extend(file_issues)
                    file_count += 1
        
        # Generate summary
        summary = self._generate_project_summary(all_issues, file_count)
        
        # Generate report
        self.generate_report(all_issues, f"{project_path}/ml_lint_report.html")
        
        return {
            "issues": all_issues,
            "summary": summary,
            "files_analyzed": file_count
        }
    
    # Helper methods
    def _generate_issue_id(self, file_path: str, line_num: int, pattern: str) -> str:
        """Generate unique issue ID"""
        content = f"{file_path}:{line_num}:{pattern}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _generate_explanation(self, pattern: str, line: str, category: str) -> str:
        """Generate human-readable explanation"""
        explanations = {
            "security_vulnerabilities": f"This pattern in '{line.strip()}' could lead to security vulnerabilities. The code should be reviewed for safer alternatives.",
            "performance_issues": f"The code '{line.strip()}' has performance implications. Consider the suggested optimization for better efficiency.",
            "code_smells": f"The pattern in '{line.strip()}' indicates a code smell that could affect maintainability."
        }
        return explanations.get(category, f"Issue detected in: {line.strip()}")
    
    def _generate_fix_suggestion(self, pattern: str, line: str) -> str:
        """Generate fix suggestion using ML"""
        fix_prompt = f"Suggest a fix for this code issue:\\n{line}"
        
        try:
            inputs = self.tokenizer(fix_prompt, return_tensors="pt", max_length=300, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.5,
                    do_sample=True
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.split("Suggest a fix for this code issue:")[-1].strip()
        except:
            return "Consider refactoring this code following best practices."
    
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 3) -> str:
        """Get surrounding context for an issue"""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        context_lines = lines[start:end]
        return "\\n".join(context_lines)
    
    def _extract_code_blocks(self, code: str) -> List[Dict]:
        """Extract functions and classes from code"""
        try:
            tree = ast.parse(code)
            blocks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    lines = code.split('\\n')
                    
                    # Extract the full block
                    block_lines = []
                    for i in range(start_line - 1, len(lines)):
                        line = lines[i]
                        if i == start_line - 1 or line.startswith(' ') or line.startswith('\\t') or line.strip() == '':
                            block_lines.append(line)
                        else:
                            break
                    
                    blocks.append({
                        "type": type(node).__name__,
                        "name": node.name,
                        "start_line": start_line,
                        "code": "\\n".join(block_lines)
                    })
            
            return blocks
        except:
            return [{"type": "file", "name": "full_file", "start_line": 1, "code": code[:500]}]
    
    def _parse_ml_analysis(self, file_path: str, block: Dict, analysis: str) -> List[CodeIssue]:
        """Parse ML analysis output into issues"""
        issues = []
        
        # Simple heuristic parsing - in production, this would be more sophisticated
        issue_keywords = ["error", "issue", "problem", "bug", "vulnerability", "smell", "warning"]
        
        if any(keyword in analysis.lower() for keyword in issue_keywords):
            issue = CodeIssue(
                id=self._generate_issue_id(file_path, block["start_line"], "ml_analysis"),
                file_path=file_path,
                line_number=block["start_line"],
                column=0,
                issue_type="ml_detected",
                severity="minor",
                title=f"ML-Detected Issue in {block['name']}",
                description="Machine learning model detected potential issues",
                explanation=f"ML Analysis: {analysis}",
                suggested_fix=self._extract_suggestion_from_analysis(analysis),
                confidence=0.7,
                code_snippet=block["code"][:200] + "...",
                context=f"Function/Class: {block['name']}",
                references=[],
                tags=["ml-detected", block["type"]],
                timestamp=datetime.now().isoformat()
            )
            issues.append(issue)
        
        return issues
    
    def _save_feedback_history(self):
        """Save feedback history to file"""
        with open("ml_lint_feedback.json", 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def _save_config(self):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    # Additional helper methods would be implemented here...
    def _analyze_imports(self, file_path: str, imports: List) -> List[CodeIssue]:
        """Analyze import statements"""
        return []  # Implementation would go here
    
    def _analyze_complexity(self, file_path: str, functions: List, code: str) -> List[CodeIssue]:
        """Analyze function complexity"""
        return []  # Implementation would go here
    
    def _extract_variables(self, tree) -> List:
        """Extract variable information"""
        return []  # Implementation would go here
    
    def _analyze_variables(self, file_path: str, variables: List, code: str) -> List[CodeIssue]:
        """Analyze variable usage"""
        return []  # Implementation would go here
    
    def _find_similar_feedback(self, issue: CodeIssue, developer_id: str) -> List:
        """Find similar feedback from history"""
        return []  # Implementation would go here
    
    def _calculate_acceptance_rate(self, feedback_list: List) -> float:
        """Calculate acceptance rate from feedback"""
        return 0.8  # Implementation would go here
    
    def _personalize_explanation(self, issue: CodeIssue, profile: Dict) -> str:
        """Personalize explanation based on developer profile"""
        return issue.explanation  # Implementation would go here
    
    def _update_learning_from_feedback(self, feedback: DeveloperFeedback):
        """Update learning models from feedback"""
        pass  # Implementation would go here
    
    def _generate_html_report(self, issues: List[CodeIssue]) -> str:
        """Generate HTML report"""
        return "<html><body><h1>ML Lint Report</h1></body></html>"  # Implementation would go here
    
    def _generate_project_summary(self, issues: List[CodeIssue], file_count: int) -> Dict:
        """Generate project analysis summary"""
        return {"total_issues": len(issues), "files_analyzed": file_count}
    
    def _generate_performance_fix(self, analysis: str) -> str:
        """Generate performance fix suggestion"""
        return "Consider optimizing this code for better performance"
    
    def _extract_suggestion_from_analysis(self, analysis: str) -> str:
        """Extract actionable suggestions from ML analysis"""
        return "Review and refactor based on ML suggestions"

# ================================
# CLI INTERFACE
# ================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML-Integrated Lint Agent")
    parser.add_argument("path", help="File or directory to analyze")
    parser.add_argument("--developer-id", default="default", help="Developer ID for personalization")
    parser.add_argument("--output", default="ml_lint_report.html", help="Output report file")
    parser.add_argument("--config", default="./ml_lint_config.json", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = MLLintAgent(config_path=args.config)
    
    # Analyze
    if os.path.isfile(args.path):
        issues = agent.analyze_file(args.path, args.developer_id)
        print(f"\\n📊 Analysis complete: {len(issues)} issues found")
        
        for issue in issues[:5]:  # Show top 5 issues
            print(f"\\n🔍 {issue.title}")
            print(f"   📍 {issue.file_path}:{issue.line_number}")
            print(f"   ⚠️  {issue.severity.upper()}: {issue.description}")
            print(f"   💡 {issue.suggested_fix}")
    
    elif os.path.isdir(args.path):
        result = agent.analyze_project(args.path, args.developer_id)
        print(f"\\n📊 Project analysis complete:")
        print(f"   Files analyzed: {result['files_analyzed']}")
        print(f"   Issues found: {len(result['issues'])}")
        print(f"   Report saved: {args.output}")
    
    else:
        print(f"❌ Path not found: {args.path}")

if __name__ == "__main__":
    main()
