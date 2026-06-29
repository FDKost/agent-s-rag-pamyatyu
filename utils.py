import os
from pathlib import Path
from typing import List

def read_text_files(directory: str) -> List[str]:
    """
    Read all .txt files from the specified directory and return their contents as a list of strings.
    """
    texts = []
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts
