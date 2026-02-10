import os
import re
import sys

def scan_codebase(root_dir):
    config_keys = set()
    # matches config.get('KEY') or config.get("KEY")
    pattern = re.compile(r"config\.get\(['\"]([A-Z_]+)['\"]")
    
    ignore_dirs = {'.venv', '.git', '__pycache__', 'dev', 'vis', 'checkpoints', 'data', 'models', 'profiles', 'results'}

    print(f"Scanning codebase at {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = pattern.findall(content)
                        if matches:
                            for m in matches:
                                config_keys.add(m)
                except Exception as e:
                    print(f"Could not read {path}: {e}")
    
    return config_keys

def check_readme(readme_path, keys):
    if not os.path.exists(readme_path):
        print(f"README not found at {readme_path}")
        return

    print(f"Reading README at {readme_path}...")
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    missing = []
    for key in keys:
        # Check for the key. We look for the exact key. 
        # In a markdown table it might appear as `KEY` or | KEY |.
        if key not in content:
            missing.append(key)
    
    return missing

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = os.path.join(root, "README.md")
    
    keys = scan_codebase(root)
    print(f"Found {len(keys)} unique config keys in codebase.")
    
    missing_keys = check_readme(readme, keys)
    
    if missing_keys:
        print("\n[!] The following keys are missing from README.md:")
        for k in sorted(missing_keys):
            print(f" - {k}")
        sys.exit(1)
    else:
        print("\n[+] All config keys are documented!")
        sys.exit(0)
