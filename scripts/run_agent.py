"""
Script to run the O!Store agent
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import main

if __name__ == "__main__":
    main()