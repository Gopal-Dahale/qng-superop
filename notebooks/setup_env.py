import sys
import os


def setup_environment():
    project_root = os.path.abspath("..")
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
