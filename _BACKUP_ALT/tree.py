import os

ignore = {".git", ".vscode", "__pycache__", "venv", "res"}


def print_tree(path, prefix=""):
    entries = [e for e in os.listdir(path) if e not in ignore]
    entries.sort()
    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(full_path):
            new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
            print_tree(full_path, new_prefix)


print_tree(".")
