import os
import logging
import warnings
from mace.calculators import MACECalculator
warnings.filterwarnings("ignore", message=".*torch.load received a zip file.*")  
from ase import Atoms
import copy
from tqdm import tqdm
import torch

torch.set_default_dtype(torch.float64)

class MaceCalc:
    """Handles loading MACE models and performing energy & force calculations."""

    def __init__(self, model_dir, device="cpu"):
        """
        Initializes MaceCalc with the model directory and device.

        Parameters:
        - model_dir (str): Path to the directory containing trained MACE models.
        - device (str): Device to run calculations ('cpu' or 'cuda').
        """
        self.model_dir = model_dir
        self.device = device
        self.models = self.load_models()
        self.num_models = len(self.models)  # Automatically determine number of models

        if self.num_models == 0:
            logging.error(f"No MACE models found in {self.model_dir}. Check the directory path.")

    def load_models(self):
        """
        Loads MACE models from the specified directory.

        Returns:
        - list: List of model file paths.
        """
        models = []
        extension = ".model"  # MACE model files have .model extension
        for filename in os.listdir(self.model_dir):
            if filename.endswith(extension):
                model_path = os.path.join(self.model_dir, filename)
                models.append(model_path)
        logging.info(f"Successfully loaded {len(models)} MACE models.")
        return models

    def calculate_energy_forces(self, atoms_list):
        """
        Calculates total energy and forces for each configuration using all available MACE models.
        Stores the calculated values inside atoms.info.

        Parameters:
        - atoms_list (list): List of ASE Atoms objects.

        Returns:
        - list: Modified list of ASE Atoms objects with updated "mace_energy" and "mace_forces" in atoms.info.
        """
        if not self.models:
            logging.error("No MACE models loaded. Cannot perform calculations.")
            return None

        progress_bar = tqdm(total=len(atoms_list), desc="Calculating MACE Energies & Forces")

        for i, atoms in enumerate(atoms_list):
            atoms_copy = copy.deepcopy(atoms)  # Create a copy to avoid modifying the original during calculation
            model_energies = []
            model_forces = []

            for model_path in self.models:
                try:
                    calc = MACECalculator(model_paths=[model_path], device=self.device, default_dtype="float64")
                    atoms_copy.calc = calc

                    energy = atoms_copy.get_potential_energy()
                    force = atoms_copy.get_forces()

                    model_energies.append(energy)
                    model_forces.append(force)
                except Exception as e:
                    logging.error(f"Error calculating energy/forces with model {model_path}: {e}")
                    model_energies.append(None)
                    model_forces.append(None)

            # Store results in original atoms.info
            atoms.info["mace_energy"] = model_energies if self.num_models > 1 else model_energies[0]
            atoms.info["mace_forces"] = model_forces if self.num_models > 1 else model_forces[0]

            progress_bar.update(1)  # Update progress bar

        progress_bar.close()  # Close progress bar when done

        return atoms_list
