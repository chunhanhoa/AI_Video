import os
import sys

# Change to the SimSwap directory
os.chdir(os.path.join(os.path.dirname(__file__), 'SimSwap'))

# Make sure SimSwap directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'SimSwap'))

print("Starting SimSwap application...")
print(f"Working directory: {os.getcwd()}")

# Run the app
try:
    from SimSwap.app import app
    app.run(debug=True, host='0.0.0.0', port=5000)
except ImportError:
    # Alternative method if the import fails
    os.system(f"{sys.executable} app.py")
