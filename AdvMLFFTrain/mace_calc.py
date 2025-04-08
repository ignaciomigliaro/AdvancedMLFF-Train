import os
import logging
import warnings
from ase.io import write, read
warnings.filterwarnings("ignore", message=".*torch.load received a zip file.*")
from mace.calculators import MACECalculator
import copy
from tqdm import tqdm
import torch

from AdvMLFFTrain.file_submit import Filesubmit

torch.set_default_dtype(torch.float64)

class MaceCalc:
    """Handles loading MACE models and performing energy & force calculations."""

    def __init__(self, model_dir, device="cpu", template_dir="templates", output_dir="mace_inference",strict=False):
        """
        Initializes MaceCalc with the model directory and device.

        Parameters:
        - model_dir (str): Path to the directory containing trained MACE models.
        - device (str): Device to run calculations ('cpu' or 'cuda').
        - template_dir (str): Directory for SLURM templates.
        - output_dir (str): Where to write inputs/outputs for evaluation.
        """
        self.model_dir = model_dir
        self.device = device
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.models = self.load_models(strict=strict)
        self.num_models = len(self.models)

        os.makedirs(self.output_dir, exist_ok=True)

        if self.num_models == 0:
            logging.error(f"No MACE models found in {self.model_dir}. Check the directory path.")

    def load_models(self, strict=False):
        models = []

        if strict:
            # Strict: only load model_n/model_n.model
            for subfolder in sorted(os.listdir(self.model_dir)):
                subfolder_path = os.path.join(self.model_dir, subfolder)
                if os.path.isdir(subfolder_path) and subfolder.startswith("model_"):
                    expected_file = f"{subfolder}.model"
                    full_model_path = os.path.join(subfolder_path, expected_file)
                    if os.path.exists(full_model_path):
                        models.append(full_model_path)
                    else:
                        logging.warning(f"Expected model not found: {full_model_path}")
        else:
            # Loose: grab all .model files anywhere
            for root, _, files in os.walk(self.model_dir):
                for filename in files:
                    if filename.endswith(".model"):
                        models.append(os.path.join(root, filename))

        logging.info(f"Successfully loaded {len(models)} model(s) from {self.model_dir}.")
        return models

    def calculate_energy_forces(self, atoms_list):
        """
        Direct (in-memory) energy and force calculation using loaded models.
        For CPU usage or debugging only.
        """
        if not self.models:
            logging.error("No MACE models loaded. Cannot perform calculations.")
            return None

        progress_bar = tqdm(total=len(atoms_list), desc="Calculating MACE Energies & Forces")

        for i, atoms in enumerate(atoms_list):
            atoms_copy = copy.deepcopy(atoms)
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
                    logging.error(f"Error with model {model_path}: {e}")
                    model_energies.append(None)
                    model_forces.append(None)

            atoms.info["mace_energy"] = model_energies if self.num_models > 1 else model_energies[0]
            atoms.info["mace_forces"] = model_forces if self.num_models > 1 else model_forces[0]
            progress_bar.update(1)

        progress_bar.close()
        return atoms_list

    def submit_mace_eval_jobs(self, atoms_list, xyz_name="eval_input.xyz", slurm_template="slurm_template_mace_eval.slurm"):
        """
        Submits SLURM evaluation jobs for each model and waits until completion.
        """
        input_xyz = os.path.join(self.output_dir, xyz_name)
        if not os.path.exists(input_xyz):
            write(input_xyz, atoms_list)
        
        submitter = Filesubmit(self.output_dir)
        submitted_scripts = []

        for model_index, model_path in enumerate(self.models):
            output_xyz = os.path.join(self.output_dir, f"evaluated_{xyz_name}".replace(".xyz", f"_model_{model_index}.xyz"))
            slurm_script = os.path.join(self.output_dir, f"mace_eval_model_{model_index}.slurm")

            self.create_slurm_script(
                template_name=slurm_template,
                output_path=slurm_script,
                input_file=os.path.basename(input_xyz),
                output_file=os.path.basename(output_xyz),
                model_path=model_path,
            )
            submitted_scripts.append(slurm_script)

        # Submit and wait
        logging.info(f"Submitting {len(submitted_scripts)} MACE evaluation jobs and waiting...")
        submitter.run_all_jobs()  # This blocks until all jobs complete


    def create_slurm_script(self, template_name, output_path, input_file, output_file, model_path):
        """
        Create a SLURM job script from a template.
        """
        template_path = os.path.join(self.template_dir, template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"SLURM template not found: {template_path}")

        with open(template_path, "r") as f:
            content = f.read()

        content = content.replace("{input_file}", input_file)
        content = content.replace("{output_file}", output_file)
        content = content.replace("{model_path}", model_path)
        content = content.replace("{workdir}", self.output_dir)

        with open(output_path, "w") as f:
            f.write(content)

        logging.info(f"Created SLURM script: {output_path}")

    def load_evaluated_results(self, evaluated_xyz):
        """
        Loads evaluated structures from MACE output .xyz file.

        Parameters:
            evaluated_xyz (str): Name of evaluated .xyz file.

        Returns:
            list of ASE Atoms: Atoms with MACE-calculated energy and forces.
        """
        evaluated_path = os.path.join(self.output_dir, evaluated_xyz)
        if not os.path.isfile(evaluated_path):
            raise FileNotFoundError(f"Evaluated XYZ file not found: {evaluated_path}")

        atoms_list = read(evaluated_path, index=":")
        logging.info(f"Loaded {len(atoms_list)} structures from evaluated MACE output: {evaluated_xyz}")
        return atoms_list
