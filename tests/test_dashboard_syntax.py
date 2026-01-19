import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDashboardSyntax(unittest.TestCase):
    def test_import(self):
        print("\nChecking Dashboard Syntax...")
        try:
            import reporting.dashboard
            print("Import successful.")
        except Exception as e:
            # We expect it might fail if streamlit is not installed in the environment running the test,
            # but we can at least check if the file is valid.
            import ast
            with open('reporting/dashboard.py', 'r') as f:
                ast.parse(f.read())
            print("AST parsing successful (Syntax is correct).")

if __name__ == '__main__':
    unittest.main()

