import os
import ast
def scan_project_structure(root_dir='.', output_file='project_structure.txt'):
    """
    Scans and saves the project structure to a text file, focusing on important files.
    """
    
    ignore_dirs = ['__pycache__', '.git', 'extraction_logs', 'teach_me', 'samples']
    ignore_files = [
        '.DS_Store', 
        'flask_ollama.log', 
        'mrworker@alambda.systems.coreftp', 
        'cert.jpg',
        'Roboto-Regular.ttf'
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Scanning project structure for: {os.path.abspath(root_dir)}\n")
        for root, dirs, files in os.walk(root_dir):
            # Exclude ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            # Sort directories and files for consistent output
            dirs.sort()
            files.sort()

            level = root.replace(root_dir, '').count(os.sep)
            if root == '.':
                level = 0
            else:
                # Adjust level for root paths that are not just '.'
                level = root.replace(root_dir, '').lstrip(os.sep).count(os.sep) + 1


            indent = ' ' * 4 * (level)
            
            # Don't print the root '.' directory name, just start with its contents
            if root != '.':
                f.write(f'{indent}{os.path.basename(root)}/\n')
            
            sub_indent = ' ' * 4 * (level + 1)
            for file_name in files:
                if file_name not in ignore_files:
                    f.write(f'{sub_indent}{file_name}\n')

if __name__ == '__main__':
    # Get the directory of the script
    script_dir = os.path.dirname(__file__)
    # If the script is in the root, we scan the parent, otherwise the script dir
    scan_dir = script_dir if script_dir else '.'
    output_filename = 'project_structure.txt'
    scan_project_structure(scan_dir, output_filename)
    print(f"Project structure saved to {output_filename}")
