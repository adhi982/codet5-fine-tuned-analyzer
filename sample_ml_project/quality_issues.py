
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
