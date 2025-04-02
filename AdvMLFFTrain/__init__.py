# AdvMLFFTrain/__init__.py
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to stdout (captured by SLURM)
    ]
)

CONFIG = {
    "data_path": os.getenv("DATA_PATH", "./data")
}

print("Your Project Package Loaded!")
