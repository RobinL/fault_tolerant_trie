"""
Ensure the project root is on sys.path so tests can import the local
"matcher" package without installation.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

