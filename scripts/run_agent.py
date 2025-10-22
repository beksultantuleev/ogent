"""
Script to run the O!Store agent
"""

import os

os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import main

if __name__ == "__main__":
    main()