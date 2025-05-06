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
from AdvMLFFTrain.utils import random_sampling


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
        self.dft_sampling = getattr(args, "dft_sampling", 100)
        self.use_cache = args.use_cache

        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f"Using calculator: {self.calculator}")
        logging.info(f"Device selected: {self.device}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # **Initialize MACE calculator if selected**
        if self.calculator.lower() == "mace":
            # Initial loose loading of any available models
            self.mace_calc = None


            # Ensure model directory exists
            if not os.path.isdir(self.args.model_dir):
                raise ValueError(f"Model directory {self.args.model_dir} does not exist.")

            # Require 3 models for active learning
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
        Run MACE evaluation for all models using SLURM, or load existing results if available.
        Returns: [atoms_list_model_0, atoms_list_model_1, atoms_list_model_2], num_structures
        """
        input_xyz = f"eval_input_iter_{iteration}.xyz"
        input_xyz_path = os.path.join(self.mace_calc.output_dir, input_xyz)

        evaluated_files = []
        output_files_exist = True

        for model_index in range(self.mace_calc.num_models):
            fname = f"evaluated_eval_input_iter_{iteration}_model_{model_index}.xyz"
            fpath = os.path.join(self.mace_calc.output_dir, fname)
            evaluated_files.append(fname)
            if not os.path.exists(fpath):
                output_files_exist = False

        if self.use_cache and output_files_exist:
            logging.info(f"[CACHE] Using cached MACE results for iteration {iteration}.")
        else:
            if self.use_cache and not output_files_exist:
                logging.warning(f"[CACHE] Expected MACE output files not found for iteration {iteration}. Re-running inference.")
            logging.info(f"AL Iteration {iteration}: Preparing SLURM jobs to evaluate {len(sampled_atoms)} structures.")
            self.mace_calc.submit_mace_eval_jobs(
                atoms_list=sampled_atoms,
                xyz_name=input_xyz,
                slurm_template="slurm_template_mace_eval.slurm"
            )

        # Load results per model
        all_atoms_lists = []
        for fname in evaluated_files:
            atoms_list = self.mace_calc.load_evaluated_results(fname)
            all_atoms_lists.append(atoms_list)

        return all_atoms_lists, len(all_atoms_lists[0])

    def sample_top_deviation_structures(self, atoms_list, std_devs):
        """
        Sorts atoms by standard deviation descending and selects top N% for DFT.

        Returns:
            sampled_atoms (list): Selected atoms for DFT.
            remaining_atoms (list): Rest to be kept in active pool.
        """
        if not atoms_list or not std_devs:
            return [], []

        # Zip and sort by deviation descending
        atoms_with_devs = list(zip(atoms_list, std_devs))
        atoms_with_devs.sort(key=lambda x: x[1], reverse=True)

        # Compute how many to keep
        n_total = len(atoms_with_devs)
        n_sample = max(1, int(n_total * self.dft_sampling / 100))

        # Split
        sampled_atoms = [x[0] for x in atoms_with_devs[:n_sample]]
        remaining_atoms = [x[0] for x in atoms_with_devs[n_sample:]]

        logging.info(f"Sampling top {self.dft_sampling}%: selected {len(sampled_atoms)}, deferred {len(remaining_atoms)}")

        return sampled_atoms, remaining_atoms

    def calculate_std_dev(self, atoms_lists_per_model):
        """
        Calculate standard deviation of energies and forces using Query by Committee.
        Input is a list of atoms_lists, one per model.
        """
        logging.info("Calculating standard deviations from multiple models.")

        if not atoms_lists_per_model or len(atoms_lists_per_model) < 2:
            logging.error("Need multiple model outputs to compute standard deviation.")
            return None, None

        n_structures = len(atoms_lists_per_model[0])
        n_models = len(atoms_lists_per_model)

        std_energy = []
        std_dev_forces = []

        progress = tqdm(total=n_structures, desc="Computing Std. Dev.")

        for i in range(n_structures):
            energy_values = []
            force_values = []

            for model_index in range(n_models):
                atoms = atoms_lists_per_model[model_index][i]
                energy_values.append(atoms.info.get("MACE_energy"))
                force_values.append(atoms.arrays.get("MACE_forces"))

            std_energy.append(np.std(energy_values))
            std_dev_forces.append(np.mean(np.std(force_values, axis=0)))

            progress.update(1)

        progress.close()

        logging.info("Standard deviations calculated.")
        logging.info(f"Mean standard deviation of forces: {np.mean(std_dev_forces)}")
        logging.info(f"Mean standard deviation of energies: {np.mean(std_energy)}")
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

    def generate_dft_inputs(self, atoms_list, iteration=0):
        if self.use_cache:
            logging.info("Skipping DFT input generation because --use_cache is set.")
            return

        dft_input = DFTInputGenerator(
            output_dir=self.output_dir,
            dft_software=self.dft_software,
            template_dir=self.template_dir
        )
        dft_input.generate_dft_inputs(atoms_list, iteration)

    def launch_dft_calcs(self):
        if self.use_cache:
            logging.info("Skipping DFT job submission because --use_cache is set.")
            return

        logging.info(f"Launching calculations in {self.output_dir}")
        cwd = os.getcwd()
        os.chdir(self.output_dir)

        try:
            submitter = Filesubmit(job_dir=".")
            submitter.run_all_jobs(max_concurrent=15)
        finally:
            os.chdir(cwd)

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
        logging.info(f"Parsing training data from {self.training_data}...")
        parser = DFTOutputParser(output_dir=self.training_data, dft_software=self.dft_software)
        return parser.parse_outputs()

    def mlff_train(self, atoms_list, iteration):
        """
        Wrap MLFFTrain class to train models with given atoms and per-iteration output_dir.
        """
        iter_output_dir = os.path.join(self.output_dir, f"models_iter_{iteration}")

        trainer = MLFFTrain(
            atoms_list=atoms_list,
            method=self.calculator,
            output_dir=iter_output_dir,
            template_dir=self.template_dir
        )

        n_models = getattr(self.mace_calc, "num_models", 1)
        trainer.prepare_and_submit_mlff(n_models=n_models)

        # Adjusted to point to the actual subdirectory where models are stored
        strict_model_dir = os.path.join(iter_output_dir, "models")

        self.mace_calc = MaceCalc(
            model_dir=strict_model_dir,
            device=self.device,
            strict=True
        )

    def get_last_completed_iteration(self):
        """
        Returns the last iteration index for which evaluated MACE inference files exist.
        """
        max_checked = 50  # Hard cap to avoid scanning forever
        for i in range(1, max_checked):
            all_exist = True
            for model_idx in range(self.mace_calc.num_models):
                fname = f"evaluated_eval_input_iter_{i}_model_{model_idx}.xyz"
                fpath = os.path.join(self.mace_calc.output_dir, fname)
                if not os.path.exists(fpath):
                    all_exist = False
                    break
            if not all_exist:
                return i - 1  # Previous iteration was the last completed
        return max_checked  # Fallback

    def dft_outputs_exist(self, iteration, num_expected):
        """
        Check whether expected DFT output files exist for a given iteration.
        """
        dft_suffix = ".out" if self.dft_software == "orca" else ".pw.out"
        matched = 0
        for fname in os.listdir(self.output_dir):
            if fname.startswith(f"iter_{iteration}_") and fname.endswith(dft_suffix):
                matched += 1
        return matched >= num_expected

    def run(self, max_iterations=10):
        """
        Executes the Active Learning pipeline iteratively.
        Stops early if no high-uncertainty structures are found.

        Parameters:
        - max_iterations (int): Maximum number of active learning iterations.
        """
        # === Load initial dataset ===
        sampled_atoms, remaining_atoms = self.load_data()

        start_iter = self.get_last_completed_iteration() + 1 if self.use_cache else 0
        if start_iter > max_iterations:
            logging.info(f"All {max_iterations} iterations already completed. Exiting.")
            return

        for iteration in range(start_iter, max_iterations + 1):
            logging.info(f"\nActive Learning Iteration {iteration}/{max_iterations}")

            # === STEP 1: Decide what to sample ===
            if iteration == 1:
                inference_candidates = sampled_atoms
            else:
                inference_candidates, remaining_atoms = random_sampling(remaining_atoms, self.sample_percentage)
                if not inference_candidates:
                    logging.info("No remaining structures to sample. Ending AL loop.")
                    break

            # === STEP 2: Run SLURM-based MACE inference on candidates ===
            sampled_atoms_model_lists, num_candidates = self.calculate_energies_forces(
                inference_candidates, iteration
            )

            # === STEP 3: Calculate standard deviation (query by committee) ===
            std_dev, std_dev_forces = self.calculate_std_dev(sampled_atoms_model_lists)

            # === STEP 4: Filter high-uncertainty structures ===
            filtered_atoms_list = self.filter_high_deviation_structures(
                std_dev,
                std_dev_forces,
                sampled_atoms_model_lists[0],  # Use model_0 atoms for reference
                percentile=90
            )

            if not filtered_atoms_list:
                logging.info("No new high-uncertainty structures found. Stopping AL loop.")
                break

            # === STEP 5: Sample top deviation structures BEFORE DFT ===
            if self.eval_criteria == "forces":
                deviations = std_dev_forces
            else:
                deviations = std_dev

            # These correspond to filtered_atoms_list
            sampled_for_dft, deferred = self.sample_top_deviation_structures(filtered_atoms_list, deviations)

            # === STEP 6: Run DFT on selected high-uncertainty structures ===
            if self.use_cache and self.dft_outputs_exist(iteration, len(sampled_for_dft)):
                logging.info(f"[CACHE] Found all expected DFT outputs for iteration {iteration}. Skipping DFT.")
            else:
                self.generate_dft_inputs(sampled_for_dft, iteration)
                self.launch_dft_calcs()

            # === STEP 7: Parse new DFT results and combine with training set ===
            new_atoms = self.parse_outputs()
            logging.info(f"Parsed {len(new_atoms)} new DFT results.")
            training_atoms = self.parse_training_data()
            all_atoms = new_atoms + training_atoms

            # === STEP 8: Retrain MLFF models ===
            model_dir = os.path.join(self.output_dir, f"models_iter_{iteration}")
            self.mlff_train(all_atoms, iteration=iteration)

            # === STEP 9: Reload updated models ===
            model_dir = os.path.join(self.output_dir, f"models_iter_{iteration}", "models")
            self.mace_calc = MaceCalc(model_dir=model_dir, device=self.device)
            self.mace_calc.models = self.mace_calc.load_models(strict=(iteration > 1))

            # === STEP 10: Update active pools ===
            sampled_atoms += sampled_for_dft
            remaining_atoms = [a for a in remaining_atoms if a not in sampled_for_dft]
            remaining_atoms += deferred  # Add non-sampled back to active pool

            logging.info("\nActive Learning process completed.")

