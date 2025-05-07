import os
import logging
from ase.io import write
from sklearn.model_selection import train_test_split
from AdvMLFFTrain.file_submit import Filesubmit  # adjust path if needed
import yaml
import subprocess
from ase.io.extxyz import write_xyz
import time

class MLFFTrain:
    """
    Handles the preprocessing of training data for different ML force field formats.
    Currently supports MACE.
    """

    def __init__(self, atoms_list, method="mace", output_dir="models",template_dir="templates"):
        """
        Parameters:
        - atoms_list (list of ASE Atoms): Training structures
        - method (str): The MLFF to format data for (e.g., 'mace', 'chgnet')
        - output_dir (str): Directory to write training data
        """
        self.atoms_list = atoms_list
        self.method = method.lower()
        self.output_dir = output_dir
        self.template_dir = template_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_and_submit_mlff(self, n_models=1):
        """
        Prepares training data and submits one or more MLFF training jobs.
        For multiple models, unique directories and config names are used.

        Parameters:
        - n_models (int): Number of models to train (for committee/active learning)
        """
        if self.method != "mace":
            raise NotImplementedError(f"MLFF method '{self.method}' is not implemented yet.")

        files = self._write_mace_xyz_split()

        job_ids = []
        for i in range(n_models):
            model_name = f"model_{i}"
            yaml_file = self.create_mace_yaml(
                train_file=files["train_file"],
                test_file=files["test_file"],
                yaml_filename=f"mace_input_{i}.yaml",
                model_name=model_name
            )
            logging.info(f"Submitting training job for {model_name} with config {yaml_file}")
            job_id = self.submit_training_job(yaml_file)
            if job_id:
                job_ids.append(job_id)

        # Wait for SLURM jobs to complete
        submitter = Filesubmit(job_dir=self.template_dir)
        submitter.run_all_jobs()

        # After jobs complete, retry-check for stage two models
        model_dir = os.path.join(self.output_dir, "models")
        expected_models = [
            os.path.join(model_dir, f"model_{i}", f"model_{i}_stagetwo.model")
            for i in range(n_models)
        ]

        max_attempts = 3
        retry_wait = 120  # seconds

        for attempt in range(1, max_attempts + 1):
            missing = [m for m in expected_models if not os.path.exists(m)]
            if not missing:
                logging.info(f"✅ All {n_models} stage two models successfully found in {model_dir}.")
                break
            else:
                if attempt < max_attempts:
                    logging.warning(
                        f"[Attempt {attempt}] Missing {len(missing)} model(s). Retrying in {retry_wait}s..."
                    )
                    time.sleep(retry_wait)
                else:
                    raise FileNotFoundError(
                        "Training completed but the following stage two model files are still missing after multiple attempts:\n" +
                        "\n".join(missing)
                    )

    def _write_mace_xyz_split(self):
        """
        Writes properly formatted train/test XYZ files for MACE.
        """
        train_data, test_data = train_test_split(self.atoms_list, test_size=0.1, random_state=42)
        logging.info(f"Total: {len(self.atoms_list)} | Train: {len(train_data)} | Test: {len(test_data)}")

        train_file = os.path.join(self.output_dir, "train.xyz")
        test_file = os.path.join(self.output_dir, "test.xyz")

        logging.info(f"Writing train to {train_file}")
        write_xyz(open(train_file, 'w'), train_data)

        logging.info(f"Writing test to {test_file}")
        write_xyz(open(test_file, 'w'), test_data)

        return {"train_file": train_file, "test_file": test_file}
    
    def create_mace_yaml(self, train_file, test_file, yaml_filename="mace_input.yaml", model_name="mace_model", template_file="mace_template.yaml"):
        """
        Creates a MACE YAML configuration file by loading a template and updating key values.
        The model_dir, log_dir, and checkpoints_dir are hardcoded subdirectories inside output_dir.

        Parameters:
        - train_file (str): Path to the training XYZ file.
        - test_file (str): Path to the test XYZ file.
        - yaml_filename (str): Name of the new YAML file to be created.
        - model_name (str): Name of the model.
        - template_file (str): Template YAML file to load from template_dir.

        Returns:
        - str: Path to the created YAML config file.
        """

        template_path = os.path.join(self.template_dir, template_file)
        yaml_path = os.path.join(self.template_dir, yaml_filename)

        if not os.path.exists(template_path):
            logging.error(f"Template file not found: {template_path}")
            return None

        with open(template_path, "r") as f:
            config = yaml.safe_load(f)

        # Define hardcoded subdirectories
        model_dir = os.path.join(self.output_dir, "models", model_name)
        log_dir = os.path.join(self.output_dir, "logs", model_name)
        checkpoints_dir = os.path.join(self.output_dir, "checkpoints", model_name)

        # Make sure those subdirectories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Update key values, hardcoding model paths
        config.update({
            "name": model_name,
            "train_file": train_file,
            "test_file": test_file,
            "model_dir": model_dir,
            "log_dir": log_dir,
            "checkpoints_dir": checkpoints_dir,
        })

        os.makedirs(self.template_dir, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        logging.info(f"Created MACE config from template: {yaml_path}")
        return yaml_path

    def submit_training_job(self, yaml_path, slurm_name="mlff_train.slurm"):
        """
        Submit SLURM job using the YAML config and SLURM script from template_dir.
        Uses Filesubmit._submit_job but ensures cwd is template_dir during sbatch call.
        """
        slurm_script_path = os.path.join(self.template_dir, slurm_name)

        if not os.path.exists(yaml_path):
            logging.error(f"YAML config not found: {yaml_path}")
            return None
        if not os.path.exists(slurm_script_path):
            logging.error(f"SLURM script not found: {slurm_script_path}")
            return None

        submitter = Filesubmit(job_dir=self.template_dir)

        # Use template_dir as the working directory
        cwd = os.getcwd()
        os.chdir(self.template_dir)

        try:
            logging.info(f"Submitting SLURM job: {slurm_script_path} with config {yaml_path}")
            job_id = submitter._submit_job(slurm_script_path, yaml_path)
            return job_id
        except Exception as e:
            logging.error(f"Submission failed: {e}")
            return None
        finally:
            os.chdir(cwd)


    
