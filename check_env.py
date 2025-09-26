import sys
import os

print("--- Python Interpreter Information ---")
print(f"Executable Path: {sys.executable}")
print(f"Version: {sys.version}")
print("\n--- sys.path (where Python looks for modules) ---")
for path in sys.path:
    print(path)

print("\n--- Checking for 'google-generativeai' installation ---")
try:
    import google.generativeai
    print("\nSUCCESS: 'google.generativeai' module was found by this interpreter.")
    print(f"Location: {google.generativeai.__file__}")
except ImportError:
    print("\nFAILURE: 'google.generativeai' module was NOT found by this interpreter.")

print("\n--- Done ---")