import sys
import os

# Set root directory in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import main function from app.dashboard
from app.dashboard import main

if __name__ == "__main__":
    main()
