import re

def modify_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Replace validate_quantization_inputs method
    content = re.sub(
        r'def validate_quantization_inputs\(self\):.*?if not self\.backend_input\.text\(\):.*?raise ValueError\("Backend path is required"\)',
        'def validate_quantization_inputs(self):\n        if not self.backend_combo.currentData():\n            raise ValueError("No backend selected")',
        content, flags=re.DOTALL
    )

    # Replace in generate_imatrix method
    content = re.sub(
        r'backend_path = self\.backend_input\.text\(\)',
        'backend_path = self.backend_combo.currentData()',
        content
    )

    # Remove browse_backend method
    content = re.sub(
        r'def browse_backend\(self\):.*?ensure_directory\(backend_path\)\n',
        '',
        content, flags=re.DOTALL
    )

    # Write the modified content back to the file
    with open(filename, 'w') as file:
        file.write(content)

    print(f"File {filename} has been modified.")

# Use the function
modify_file('AutoGGUF.py')
