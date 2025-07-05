
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
