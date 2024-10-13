#!/bin/sh

# Check if Python is installed
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: Python 3 is not installed or not in the PATH."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Set environment variables
export PYTHONIOENCODING=utf-8
export AUTOGGUF_LANGUAGE=en-US

# Try to run main.py in the current directory
if [ -f "main.py" ]; then
    echo "Running main.py in the current directory..."
    python3 main.py
    exit 0
fi

# If main.py doesn't exist in the current directory, try src/main.py
if [ -f "src/main.py" ]; then
    echo "Running src/main.py..."
    python3 src/main.py
    exit 0
fi

# If neither file is found, display an error message
echo "Error: Neither main.py nor src/main.py found."
echo "Please make sure the script is in the correct directory."
exit 1
