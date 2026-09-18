import os
import yaml
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_loader")

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(file_path: str = None, config: dict = None) -> pd.DataFrame:
    """Loads raw Excel dataset and parses datetime columns."""
    if config is None:
        config = load_config()
    
    path = file_path or config.get("raw_data_path", "Research Project.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at {path}")
    
    logger.info(f"Loading raw data from {path}...")
    df = pd.read_excel(path)
    
    # Parse datetimes
    from_col = config["datetime_columns"]["from_date"]
    if from_col in df.columns:
        df["datetime"] = pd.to_datetime(df[from_col], format="%d-%m-%Y %H:%M", errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df
