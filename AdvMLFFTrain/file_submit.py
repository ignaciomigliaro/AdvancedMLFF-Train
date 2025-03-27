import os
import subprocess
import logging
import time
from pathlib import Path
import getpass

class Filesubmit:
    def __init__(self, job_dir, slurm_ext=".slurm"):
        """
        Parameters:
        - job_dir (str): Directory containing SLURM job scripts.
        - slurm_ext (str): Extension to identify SLURM scripts.
        """
        self.job_dir = job_dir
        self.slurm_ext = slurm_ext

    def _get_running_job_ids(self):
        """
        Get a set of running SLURM job IDs for the current user.
        """
        try:
            username = getpass.getuser()
            output = subprocess.check_output(
                f"squeue -u {username} -h -o %i",
                shell=True,
                text=True
            )
            return set(output.strip().splitlines())
        except Exception as e:
            logging.error(f"Error retrieving SLURM jobs: {e}")
            return set()

    def _submit_job(self, script_path):
        """
        Submit a single SLURM job and return its job ID.
        """
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"Submitted {script_path} → Job ID: {job_id}")
            return job_id
        else:
            logging.error(f"Failed to submit {script_path}: {result.stderr}")
            return None

    def _find_slurm_scripts(self):
        """
        Find all SLURM scripts in the job_dir with the given extension.
        """
        return sorted([
            os.path.join(self.job_dir, f)
            for f in os.listdir(self.job_dir)
            if f.endswith(self.slurm_ext)
        ])

    def run_all_jobs(self, max_concurrent=4, sleep_interval=30):
        """
        Submit and monitor SLURM jobs with concurrency control.

        Parameters:
        - max_concurrent (int): Max number of jobs allowed in the queue at once.
        - sleep_interval (int): Time (seconds) to wait between queue checks.
        """
        logging.info(f"Looking for SLURM scripts in: {self.job_dir}")
        scripts_to_submit = self._find_slurm_scripts()
        logging.info(f"Found {len(scripts_to_submit)} SLURM scripts to submit.")

        if not scripts_to_submit:
            logging.warning("No SLURM job scripts found. Nothing to submit.")
            return

        submitted_job_ids = []

        for script in scripts_to_submit:
            while True:
                running_jobs = self._get_running_job_ids()
                if len(running_jobs) < max_concurrent:
                    job_id = self._submit_job(script)
                    if job_id:
                        submitted_job_ids.append(job_id)
                    break
                else:
                    logging.info(
                        f"[WAITING] {len(running_jobs)} jobs active. Limit: {max_concurrent}. Sleeping..."
                    )
                    time.sleep(sleep_interval)

        logging.info("All jobs submitted. Monitoring until completion...")
        while True:
            active = self._get_running_job_ids()
            still_running = [jid for jid in submitted_job_ids if jid in active]
            if still_running:
                logging.info(f"Jobs still running: {', '.join(still_running)}")
                time.sleep(sleep_interval)
            else:
                logging.info("✅ All submitted jobs have completed.")
                break
