import sys
import os
import subprocess

# Ensure current project directory is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.logger import get_logger
from src.models.train import train_and_evaluate_models

logger = get_logger("cli_launcher")

def print_help():
    print("""
========================================================================
🌫️ Air Quality ML System - CLI Launcher
========================================================================
Usage: python main.py [command]

Available Commands:
  train       : Run data preprocessing, feature engineering, and model training
  api         : Launch FastAPI REST server (http://127.0.0.1:8000/docs)
  dashboard   : Launch Streamlit Web Dashboard (http://localhost:8501)
  test        : Run Pytest automated test suite
  help        : Show this usage guide
========================================================================
""")

def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)
        
    cmd = sys.argv[1].lower()
    
    if cmd == "train":
        logger.info("Executing training pipeline...")
        train_and_evaluate_models()
    elif cmd == "api":
        logger.info("Starting FastAPI Uvicorn server...")
        subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
    elif cmd == "dashboard":
        logger.info("Launching Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])
    elif cmd == "test":
        logger.info("Running Pytest suite...")
        subprocess.run([sys.executable, "-m", "pytest", "-v"])
    else:
        print_help()

if __name__ == "__main__":
    main()
