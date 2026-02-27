#!/usr/bin/env python3
"""
Concatenates all .txt files from the context directory into a single text file.
Each file's content is prefixed with its filename for source traceability.
"""

import os
import glob
import sys

CONTEXT_DIR = os.path.join(os.path.dirname(__file__), "..", "context")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "full_context.txt")


def concat_context(context_dir: str = CONTEXT_DIR, output_file: str = OUTPUT_FILE) -> str:
    """Concatenate all .txt files from context_dir into output_file."""
    txt_files = sorted(glob.glob(os.path.join(context_dir, "*.txt")))

    if not txt_files:
        print(f"No .txt files found in {context_dir}")
        sys.exit(1)

    total_chars = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for i, filepath in enumerate(txt_files):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                print(f"  Warning: Could not read {filename}: {e}")
                continue

            # Write document header and content
            out.write(f"\n{'='*80}\n")
            out.write(f"[DOCUMENT: {filename}]\n")
            out.write(f"{'='*80}\n")
            out.write(content)
            out.write("\n")
            total_chars += len(content)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(txt_files)} files...")

    print(f"\nDone: {len(txt_files)} files concatenated")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: ~{total_chars // 4:,}")
    print(f"Output: {output_file}")
    return output_file


if __name__ == "__main__":
    concat_context()
