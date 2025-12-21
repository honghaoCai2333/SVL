"""
Script to migrate print statements to logger in Python files
"""

import re
import sys
from pathlib import Path

def add_logger_import(content: str, file_path: str) -> str:
    """Add logger import if not present"""

    # Check if logger import already exists
    if 'from src.utils.logger import get_logger' in content:
        return content

    # Find the last import statement
    lines = content.split('\n')
    last_import_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i
        elif stripped and not stripped.startswith('#') and last_import_idx != -1:
            # Found first non-import, non-comment line after imports
            break

    if last_import_idx == -1:
        # No imports found, add at the beginning
        lines.insert(0, 'from src.utils.logger import get_logger\n\nlogger = get_logger(__name__)\n')
    else:
        # Add after last import
        lines.insert(last_import_idx + 1, '\nfrom src.utils.logger import get_logger\n\nlogger = get_logger(__name__)\n')

    return '\n'.join(lines)


def replace_print_statements(content: str) -> str:
    """Replace print() with logger calls"""

    # Pattern to match print statements
    # This handles: print("..."), print(f"..."), print(variable), etc.

    # Replace print(...) with logger.info(...)
    content = re.sub(r'\bprint\s*\(', 'logger.info(', content)

    return content


def process_file(file_path: Path):
    """Process a single file"""
    print(f"Processing {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add logger import
        content = add_logger_import(content, str(file_path))

        # Replace print statements
        content = replace_print_statements(content)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✓ Done")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    files_to_process = [
        'src/data/dataset.py',
        'src/data/negative_sampler.py',
        'scripts/prepare_data.py',
        'data_build.py',
        'src/data_generation/generate_step1.py',
        'src/data_generation/generate_step2.py',
        'src/utils/state_machine.py',
        'src/utils/prompt_template.py',
    ]

    base_dir = Path(__file__).parent.parent

    for file_rel in files_to_process:
        file_path = base_dir / file_rel
        if file_path.exists():
            process_file(file_path)
        else:
            print(f"File not found: {file_path}")


if __name__ == '__main__':
    main()
