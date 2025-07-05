"""
ML LINT AGENT INTEGRATION
Ready-to-use integration examples for different development workflows

This shows how to integrate the ML Lint Agent into:
- Git hooks (pre-commit)
- CI/CD pipelines
- IDE plugins
- Code review processes
- Development workflows
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_lint_agent import MLLintAgent

class GitHookIntegration:
    """Integration with Git hooks"""
    
    def __init__(self):
        self.agent = MLLintAgent()
    
    def pre_commit_hook(self):
        """Pre-commit hook to analyze changed files"""
        print("🔍 ML Lint Agent - Pre-commit Analysis")
        
        # Get staged files
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                capture_output=True, text=True
            )
            staged_files = [f for f in result.stdout.strip().split('\\n') if f.endswith('.py')]
        except:
            print("⚠️  Could not get staged files")
            return 0
        
        if not staged_files:
            print("✅ No Python files to analyze")
            return 0
        
        print(f"📁 Analyzing {len(staged_files)} Python files...")
        
        critical_issues = []
        total_issues = 0
        
        for file_path in staged_files:
            if os.path.exists(file_path):
                issues = self.agent.analyze_file(file_path, developer_id="git_user")
                total_issues += len(issues)
                
                # Check for critical issues
                critical = [i for i in issues if i.severity == 'critical']
                if critical:
                    critical_issues.extend(critical)
                    print(f"🚨 {file_path}: {len(critical)} critical issues")
                else:
                    print(f"✅ {file_path}: {len(issues)} issues (no critical)")
        
        print(f"\\n📊 Analysis Summary:")
        print(f"   Total issues: {total_issues}")
        print(f"   Critical issues: {len(critical_issues)}")
        
        if critical_issues:
            print("\\n🚨 Critical Issues Found:")
            for issue in critical_issues[:5]:  # Show top 5
                print(f"   • {issue.file_path}:{issue.line_number} - {issue.title}")
                print(f"     Fix: {issue.suggested_fix}")
            
            print("\\n❌ Commit blocked due to critical issues")
            print("💡 Fix critical issues and try again")
            return 1  # Block commit
        
        print("\\n✅ No critical issues found. Commit allowed.")
        return 0  # Allow commit

class CIPipelineIntegration:
    """Integration with CI/CD pipelines"""
    
    def __init__(self):
        self.agent = MLLintAgent()
    
    def analyze_pull_request(self, base_branch="main"):
        """Analyze files changed in a pull request"""
        print("🔍 ML Lint Agent - Pull Request Analysis")
        
        # Get changed files in PR
        try:
            result = subprocess.run(
                ["git", "diff", f"{base_branch}..HEAD", "--name-only"],
                capture_output=True, text=True
            )
            changed_files = [f for f in result.stdout.strip().split('\\n') if f.endswith('.py')]
        except:
            print("⚠️  Could not get changed files")
            return {"status": "error", "message": "Failed to get changed files"}
        
        if not changed_files:
            return {"status": "success", "message": "No Python files changed"}
        
        print(f"📁 Analyzing {len(changed_files)} changed Python files...")
        
        all_issues = []
        for file_path in changed_files:
            if os.path.exists(file_path):
                issues = self.agent.analyze_file(file_path, developer_id="ci_bot")
                all_issues.extend(issues)
        
        # Generate CI report
        report = self._generate_ci_report(all_issues)
        
        # Save report for CI system
        with open("ml_lint_ci_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_ci_report(self, issues):
        """Generate CI-friendly report"""
        critical_count = len([i for i in issues if i.severity == 'critical'])
        major_count = len([i for i in issues if i.severity == 'major'])
        minor_count = len([i for i in issues if i.severity == 'minor'])
        
        status = "success"
        if critical_count > 0:
            status = "failure"
        elif major_count > 5:  # Configurable threshold
            status = "warning"
        
        return {
            "status": status,
            "total_issues": len(issues),
            "critical_issues": critical_count,
            "major_issues": major_count,
            "minor_issues": minor_count,
            "issues": [
                {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "severity": issue.severity,
                    "title": issue.title,
                    "description": issue.description,
                    "fix": issue.suggested_fix
                }
                for issue in issues[:20]  # Limit for CI
            ]
        }

class IDEIntegration:
    """Integration with IDEs and editors"""
    
    def __init__(self):
        self.agent = MLLintAgent()
    
    def analyze_on_save(self, file_path, developer_id="ide_user"):
        """Analyze file when saved in IDE"""
        if not file_path.endswith('.py'):
            return []
        
        issues = self.agent.analyze_file(file_path, developer_id)
        
        # Convert to IDE-friendly format (LSP-like)
        diagnostics = []
        for issue in issues:
            diagnostics.append({
                "range": {
                    "start": {"line": issue.line_number - 1, "character": issue.column},
                    "end": {"line": issue.line_number - 1, "character": issue.column + 10}
                },
                "severity": self._severity_to_lsp(issue.severity),
                "message": f"{issue.title}: {issue.description}",
                "source": "ml-lint-agent",
                "code": issue.id,
                "codeDescription": {
                    "href": f"https://ml-lint-docs.com/issues/{issue.issue_type}"
                }
            })
        
        return diagnostics
    
    def _severity_to_lsp(self, severity):
        """Convert severity to LSP diagnostic severity"""
        mapping = {
            "critical": 1,  # Error
            "major": 1,     # Error  
            "minor": 2,     # Warning
            "suggestion": 3 # Information
        }
        return mapping.get(severity, 2)

class CodeReviewIntegration:
    """Integration with code review platforms"""
    
    def __init__(self):
        self.agent = MLLintAgent()
    
    def generate_review_comments(self, file_path, developer_id="reviewer_bot"):
        """Generate review comments for a file"""
        issues = self.agent.analyze_file(file_path, developer_id)
        
        comments = []
        for issue in issues:
            # Generate human-friendly review comment
            comment = self._format_review_comment(issue)
            comments.append({
                "file": issue.file_path,
                "line": issue.line_number,
                "comment": comment,
                "severity": issue.severity
            })
        
        return comments
    
    def _format_review_comment(self, issue):
        """Format issue as code review comment"""
        comment = f"## {issue.title}\\n\\n"
        comment += f"**Severity:** {issue.severity.title()}\\n"
        comment += f"**Description:** {issue.description}\\n\\n"
        comment += f"**Suggested Fix:**\\n{issue.suggested_fix}\\n\\n"
        
        if issue.references:
            comment += f"**References:**\\n"
            for ref in issue.references:
                comment += f"- {ref}\\n"
        
        comment += f"\\n*Detected by ML Lint Agent (Confidence: {issue.confidence:.0%})*"
        return comment

class DevelopmentWorkflowIntegration:
    """Integration with development workflows"""
    
    def __init__(self):
        self.agent = MLLintAgent()
    
    def daily_code_quality_report(self, project_path):
        """Generate daily code quality report"""
        print("📊 Generating Daily Code Quality Report...")
        
        result = self.agent.analyze_project(project_path, developer_id="daily_bot")
        
        # Generate summary
        issues_by_severity = {}
        issues_by_type = {}
        files_with_issues = set()
        
        for issue in result['issues']:
            # Count by severity
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
            
            # Count by type
            issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1
            
            # Track files with issues
            files_with_issues.add(issue.file_path)
        
        # Calculate quality score
        total_files = result['files_analyzed']
        clean_files = total_files - len(files_with_issues)
        quality_score = (clean_files / total_files * 100) if total_files > 0 else 100
        
        report = {
            "date": "2025-07-04",  # Would be dynamic
            "project_path": project_path,
            "quality_score": round(quality_score, 1),
            "total_files": total_files,
            "files_with_issues": len(files_with_issues),
            "total_issues": len(result['issues']),
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
            "top_issues": [
                {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "title": issue.title,
                    "severity": issue.severity
                }
                for issue in sorted(result['issues'], key=lambda x: x.priority_score, reverse=True)[:10]
            ]
        }
        
        # Save report
        with open(f"daily_quality_report_{project_path.replace('/', '_')}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved. Quality Score: {quality_score:.1f}%")
        return report

def create_git_pre_commit_hook():
    """Create a pre-commit hook script"""
    hook_script = '''#!/usr/bin/env python3
"""
Git pre-commit hook using ML Lint Agent
"""

import sys
import os

# Add the path to your ML Lint Agent
sys.path.append('/path/to/your/ml_lint_agent')

from ml_lint_integration import GitHookIntegration

def main():
    integration = GitHookIntegration()
    return integration.pre_commit_hook()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
'''
    
    # Save hook script
    with open("pre-commit-ml-lint", "w") as f:
        f.write(hook_script)
    
    print("✅ Pre-commit hook script created: pre-commit-ml-lint")
    print("💡 To install: copy to .git/hooks/pre-commit and make executable")

def create_ci_pipeline_config():
    """Create CI pipeline configuration examples"""
    
    # GitHub Actions workflow
    github_workflow = '''name: ML Lint Analysis

on:
  pull_request:
    branches: [ main, develop ]

jobs:
  ml-lint:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install torch transformers accelerate
        # Install your ML Lint Agent
    
    - name: Run ML Lint Analysis
      run: |
        python -c "
        from ml_lint_integration import CIPipelineIntegration
        ci = CIPipelineIntegration()
        result = ci.analyze_pull_request()
        if result['status'] == 'failure':
            exit(1)
        "
    
    - name: Upload ML Lint Report
      uses: actions/upload-artifact@v3
      with:
        name: ml-lint-report
        path: ml_lint_ci_report.json
'''
    
    with open("github_workflow_ml_lint.yml", "w") as f:
        f.write(github_workflow)
    
    print("✅ GitHub Actions workflow created: github_workflow_ml_lint.yml")

def demo_integrations():
    """Demo all integration examples"""
    print("🔗 ML LINT AGENT INTEGRATION DEMO")
    print("="*50)
    
    # Demo Git Hook Integration
    print("\\n1. 🔧 Git Hook Integration")
    git_integration = GitHookIntegration()
    print("   Pre-commit analysis ready!")
    
    # Demo CI Integration
    print("\\n2. 🚀 CI/CD Integration")
    ci_integration = CIPipelineIntegration()
    print("   Pull request analysis ready!")
    
    # Demo IDE Integration
    print("\\n3. 💻 IDE Integration")
    ide_integration = IDEIntegration()
    print("   IDE diagnostics ready!")
    
    # Demo Code Review Integration
    print("\\n4. 👥 Code Review Integration")
    review_integration = CodeReviewIntegration()
    print("   Review comments ready!")
    
    # Demo Workflow Integration
    print("\\n5. 📊 Workflow Integration")
    workflow_integration = DevelopmentWorkflowIntegration()
    print("   Daily reports ready!")
    
    print("\\n✅ All integrations ready for use!")

def main():
    """Main function"""
    print("🔗 ML LINT AGENT INTEGRATIONS")
    print("="*40)
    print("Ready-to-use integrations for your development workflow")
    
    options = [
        ("1", "Demo All Integrations", demo_integrations),
        ("2", "Create Git Pre-commit Hook", create_git_pre_commit_hook),
        ("3", "Create CI Pipeline Config", create_ci_pipeline_config),
        ("4", "Test Git Hook", lambda: GitHookIntegration().pre_commit_hook()),
        ("5", "Generate Daily Report", lambda: DevelopmentWorkflowIntegration().daily_code_quality_report(".")),
    ]
    
    print("\\n📋 Available Options:")
    for num, title, _ in options:
        print(f"   {num}. {title}")
    
    choice = input("\\n🎯 Choose option (1-5): ").strip()
    
    for num, title, func in options:
        if choice == num:
            print(f"\\n🚀 Running: {title}")
            func()
            break
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
