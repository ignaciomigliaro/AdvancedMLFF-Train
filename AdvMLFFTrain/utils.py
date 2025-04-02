import argparse
import os
import logging
from ase.io import read
import numpy as np 
import random

def parse_args():
    """Parses command-line arguments for both active learning and data reduction."""
    parser = argparse.ArgumentParser(description="Machine Learning Force Field Training")

    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    # Active Learning Pipeline
    al_parser = subparsers.add_parser("active_learning", help="Run active learning pipeline")
    al_parser.add_argument("--filepath", type=str, required=True)
    al_parser.add_argument("--model_dir", type=str, required=True)
    al_parser.add_argument("--calculator", type=str, required=True)
    al_parser.add_argument("--device", type=str, default="cpu")
    al_parser.add_argument("--threshold", type=float)
    al_parser.add_argument("--output_dir", type=str, default="results")
    al_parser.add_argument("--dft_software", type=str, choices=["qe", "orca"])
    al_parser.add_argument("--eval_criteria", type=str, choices=["energy", "forces"], default="energy")
    al_parser.add_argument("--use_cache", type=str)
    al_parser.add_argument("--plot_std_dev", action="store_true")
    al_parser.add_argument("--lower_threshold", type=float)
    al_parser.add_argument("--upper_threshold", type=float)
    al_parser.add_argument("--sample_percentage", type=int, required=True)

    # Data Reduction Pipeline
    dr_parser = subparsers.add_parser("data_reduction", help="Run data reduction pipeline")
    dr_parser.add_argument("--filepath", type=str, required=True)
    dr_parser.add_argument("--stepsize", type=int, default=1)
    dr_parser.add_argument("--model_dir", type=str, required=True)
    dr_parser.add_argument("--calculator", type=str, required=True)
    dr_parser.add_argument("--device", type=str, default="cpu")
    dr_parser.add_argument("--output_dir", type=str, default="reduced_data")
    dr_parser.add_argument("--use_cache", type=str)
    dr_parser.add_argument("--target_size", type=int, required=True)

    return parser.parse_args()

def random_sampling(atoms_list, percentage):
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100")
        num_to_sample = int(len(atoms_list) * (percentage / 100))
        sampled_atoms = random.sample(atoms_list, num_to_sample)
        remaining_atoms = [atom for atom in atoms_list if atom not in sampled_atoms]
        return sampled_atoms, remaining_atoms


def get_configurations(filepath,percentage=10):
    """Reads configurations from input files, optionally including DFT energy data."""
    logging.info(f"Reading configurations from {filepath}...")

    if os.path.isfile(filepath):
        configurations = read(filepath, index=":")
    elif os.path.isdir(filepath):
        configurations = []
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            if os.path.isfile(file_path):
                configurations.extend(read(file_path, index=":"))
    else:
        raise ValueError(f"Invalid path: {filepath}")
    sampled_atoms, remaining_atoms = random_sampling(configurations, percentage)
    return sampled_atoms, remaining_atoms


    
    