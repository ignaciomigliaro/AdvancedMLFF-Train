import logging
import torch
from AdvMLFFTrain.mace_calc import MaceCalc
from AdvMLFFTrain.dft_files import DFTInputGenerator
from AdvMLFFTrain.dft_files import DFTOutputParser
from AdvMLFFTrain.utils import get_configurations
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from AdvMLFFTrain.file_submit import Filesubmit
from AdvMLFFTrain.mlff_train import MLFFTrain

class ActiveLearning:
    """Handles the active learning pipeline for MACE MLFF models."""

    def __init__(self, args):
        """
        Initializes the Active Learning pipeline with user-defined arguments.

        Parameters:
        - args (Namespace): Parsed command-line arguments.
        """
        self.args = args
        self.device = args.device
        self.calculator = args.calculator  # MACE or another DFT calculator
        self.output_dir = args.output_dir
        self.dft_software = args.dft_software
        self.template_dir = args.template_dir if self.dft_software.lower() == "orca" else None
        self.eval_criteria = args.eval_criteria
        self.upper_threshold = args.upper_threshold
        self.lower_threshold = args.lower_threshold
        self.use_cache = args.use_cache
        self.plot_std_dev = args.plot_std_dev
        self.sample_percentage = args.sample_percentage
        self.training_data = args.training_data_dir
        self.max_al_iter = args.max_al_iter

        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f"Using calculator: {self.calculator}")
        logging.info(f"Device selected: {self.device}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # **Initialize MACE calculator if selected**
        if self.calculator.lower() == "mace":
            self.mace_calc = MaceCalc(self.args.model_dir, self.device)
            # **Explicitly check models in model_dir**
            if not os.path.isdir(self.args.model_dir):
                raise ValueError(f"Model directory {self.args.model_dir} does not exist.")

            # **Ensure at least 3 models for active learning**
            if self.mace_calc.num_models < 3:
                raise ValueError(
                    f"Active Learning requires at least 3 MACE models, but only {self.mace_calc.num_models} were found in {self.args.model_dir}. "
                    f"Check if the correct models are present."
                )

            logging.info(f"Initialized MACE calculator with {self.mace_calc.num_models} models from {self.args.model_dir}.")

    def plot_std_dev_distribution(std_devs):
        """
        Plots the distribution of standard deviations using a histogram.

        Parameters:
        - std_devs (list): List of standard deviation values to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(std_devs, bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.axvline(x=np.percentile(std_devs, 98), color='r', linestyle='--', label='98th Percentile')
        plt.legend()
        plt.grid(True)
        plt.show()

    def load_data(self):
        """Loads configurations using ASE-supported formats and initializes MACE models."""
        logging.info(f"Loading configurations from {self.args.filepath} using ASE.")

        # Load configurations using ASE-supported formats
        sampled_atoms,remaining_atoms = get_configurations(
            self.args.filepath, 
            self.args.sample_percentage, 
        )

        logging.info(f"Loaded {len(sampled_atoms)} configurations.")
        return sampled_atoms, remaining_atoms

    def calculate_energies_forces(self, sampled_atoms, iteration):
        """
        Run SLURM-based evaluation of all MACE models on the same structure set.
        Returns a list of atoms_list per model.
        """
        input_xyz = f"eval_input_iter_{iteration}.xyz"
        logging.info(f"AL Iteration {iteration}: Preparing SLURM jobs to evaluate {len(sampled_atoms)} structures.")

        for model_index in range(self.mace_calc.num_models):
            self.mace_calc.submit_mace_eval_job(
                atoms_list=sampled_atoms,
                model_index=model_index,
                job_name=f"mace_eval_iter_{iteration}_model_{model_index}",
                xyz_name=input_xyz,
                slurm_template="slurm_template_mace_eval.slurm"
            )

        # Wait for all jobs to finish
        submitter = Filesubmit(job_dir=self.mace_calc.output_dir)
        submitter.run_all_jobs(max_concurrent=1)

        # Load evaluated atoms from each model
        all_atoms_lists = []
        for model_index in range(self.mace_calc.num_models):
            evaluated_file = f"evaluated_{input_xyz}".replace(".xyz", f"_model_{model_index}.xyz")
            atoms_list = self.mace_calc.load_evaluated_results(evaluated_file)
            all_atoms_lists.append(atoms_list)

        return all_atoms_lists

    def calculate_std_dev(self, sampled_atoms):
        """
        Calculate the standard deviation of energies and forces for each atomic configuration
        in the active learning set (Query by Committee).

        Parameters:
        - cache_file (str, optional): Path to save computed energy values, forces, and deviations.

        Returns:
        - std_energy (list): Standard deviation of energies per configuration.
        - std_dev_forces (list): Standard deviation of forces per atom in each configuration.
        - energy_values (list): Computed energy values.
        - force_values (list): Computed force values.
        """
        
        logging.info("Calculating standard deviations for energies and forces.")

        if not sampled_atoms:
            logging.error("No configurations available to compute standard deviation.")
            return None, None, None, None

        std_energy = []
        std_dev_forces = []

        progress = tqdm(total=len(sampled_atoms), desc="Processing Energies and Forces")

        for atoms in sampled_atoms:
            # Extract energy values from different models
            energy_values = atoms.info["mace_energy"]  # Should be a list of 3 values (one per model)
            std_energy.append(np.std(energy_values))  # Compute standard deviation of energies

            # Extract forces from different models
            force_values = np.array(atoms.info["mace_forces"])  # Shape: (3, N_atoms, 3)

            # Compute standard deviation of forces across models for each atom
            std_dev_atom_forces = np.std(force_values, axis=0)  # Shape: (N_atoms, 3)
            std_dev_forces.append(np.mean(std_dev_atom_forces))  # Mean over all atoms and directions

            progress.update(1)

        progress.close()

        #self.std_dev = std_energy
        #self.std_dev_forces = std_dev_forces  # This is now correctly computed

        logging.info("Standard deviations calculated for all configurations.")

        return std_energy, std_dev_forces

    def filter_high_deviation_structures(self,std_dev,std_dev_forces,sampled_atoms,percentile=90):
        """
        Filters structures based on the normalized standard deviation.
        Includes structures with normalized deviation within the specified threshold range.

        Parameters:
        - atoms_lists (list of list of ASE Atoms): List containing multiple atoms lists for each model.
        - energies (list of list of floats): List containing energies for each model.
        - std_dev (list of floats): Standard deviation values.
        - user_threshold (float, optional): User-defined upper threshold for filtering. If None, percentile-based threshold is used.
        - lower_threshold (float, optional): User-defined lower threshold for filtering. If None, no lower threshold is applied.
        - percentile (int): Percentile threshold for filtering if no user threshold is provided.

        Returns:
        - filtered_atoms_list (list of ASE Atoms): List of filtered structures.
        - filtered_std_dev (list of floats): List of standard deviation values corresponding to the filtered structures.
        """
        if self.eval_criteria == 'forces':
            std_dev = std_dev_forces
        if self.eval_criteria == 'energy':
            std_dev == std_dev
        
        if self.upper_threshold and self.lower_threshold is not None:
            logging.info(f"User-defined upper threshold for filtering: {self.upper_threshold}")
        else:
            upper_threshold = np.percentile(std_dev, percentile)
            logging.info(f"Threshold for filtering (95th percentile): {percentile}")

        if self.lower_threshold is not None:
            lower_threshold = float(self.lower_threshold)
            logging.info(f"User-defined lower threshold for filtering: {lower_threshold}")
        else:
            lower_threshold = float('-inf')  # No lower threshold

        # Filter structures based on the chosen thresholds
        filtered_atoms_list = []
        filtered_std_dev = []

        for i, norm_dev in enumerate(std_dev):
            if self.lower_threshold <= norm_dev <= self.upper_threshold:  # Include structures within the threshold range
                filtered_atoms_list.append(sampled_atoms[i])
                filtered_std_dev.append(norm_dev)
        logging.info(f"Number of structures within threshold range: {len(filtered_atoms_list)}")
        return filtered_atoms_list

    def plot_std_dev_distribution(std_devs):
        """
        Plots the distribution of standard deviations using a histogram.

        Parameters:
        - std_devs (list): List of standard deviation values to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(std_devs, bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.axvline(x=np.percentile(std_devs, 98), color='r', linestyle='--', label='98th Percentile')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_dft_inputs(self, atoms_list):
            """
            Generate DFT input files for ORCA or QE and return the output directory.
            """
            dft_input = DFTInputGenerator(
                output_dir=self.output_dir, 
                dft_software=self.dft_software, 
                template_dir=self.template_dir
            )
            dft_input.generate_dft_inputs(atoms_list)

    def launch_dft_calcs(self):
        """
        Launch DFT calculations using the generated input files.
        """
        logging.info(f"Launching calculations in {self.output_dir}")
        
        # Temporarily change to input directory
        cwd = os.getcwd()
        os.chdir(self.output_dir)

        try:
            submitter = Filesubmit(job_dir=".")
            submitter.run_all_jobs(max_concurrent=15)
        finally:
            os.chdir(cwd)  # Restore original working directory

    def parse_outputs(self):
        """
        Parses DFT output files based on selected DFT software.

        Returns:
        - List of parsed results (each as dict with atoms, energy, etc.)
        """
        parser = DFTOutputParser(output_dir=self.output_dir, dft_software=self.dft_software)
        return parser.parse_outputs()
    
    def parse_training_data(self):
        """
        Parses the previous training data outputs on the selected DFT software
        """
        parser = DFTOutputParser(output_dir=self.training_data, dft_software=self.dft_software)
        return parser.parse_outputs()

    def mlff_train(self,atoms_list):
        trainer = MLFFTrain(atoms_list=atoms_list,
                  method=self.calculator,
                  output_dir=self.output_dir,
                  template_dir=self.template_dir)
        n_models = getattr(self.mace_calc, "num_models", 1)
        trainer.prepare_and_submit_mlff(n_models=n_models)

    def run(self, max_iterations=10):
        """
        Executes the Active Learning pipeline iteratively.
        Stops early if no high-uncertainty structures are found.

        Parameters:
        - max_iterations (int): Maximum number of active learning iterations.
        """
        # Initial dataset load
        sampled_atoms, remaining_atoms = self.load_data()

        for iteration in range(1, max_iterations + 1):
            logging.info(f"Active Learning Iteration {iteration}/{max_iterations}")

            # Evaluate uncertainty with current model
            sampled_atoms = self.calculate_energies_forces(sampled_atoms, iteration)
            std_dev, std_dev_forces = self.calculate_std_dev(sampled_atoms)

            # Filter structures based on uncertainty
            filtered_atoms_list = self.filter_high_deviation_structures(
                std_dev, std_dev_forces, sampled_atoms
            )

            if not filtered_atoms_list:
                logging.info("No new high-uncertainty structures found. Stopping AL loop.")
                break

            # Generate and run DFT calculations for selected structures
            self.generate_dft_inputs(filtered_atoms_list)
            self.launch_dft_calcs()

            # Parse DFT results and update training pool
            new_atoms = self.parse_outputs()
            training_atoms = self.parse_training_data()
            all_atoms = new_atoms + training_atoms

            # Update the MLFF model
            self.mlff_train(all_atoms)

            # Move newly labeled atoms to sampled_atoms
            sampled_atoms += filtered_atoms_list

            # Remove them from the pool of remaining_atoms
            remaining_atoms = [atom for atom in remaining_atoms if atom not in filtered_atoms_list]

        logging.info("Active Learning process completed.")
