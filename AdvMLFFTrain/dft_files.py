import os
import logging
from ase.io import read,write
import numpy as np 
from ase import Atoms

class DFTInputGenerator:
    """Generates DFT input files for ORCA and Quantum ESPRESSO (QE)."""

    def __init__(self, output_dir, dft_software, template_dir="templates"):
        """
        Initializes the DFTInputGenerator.

        Parameters:
        - output_dir (str): Directory where input files will be saved.
        - dft_software (str): Specifies whether to generate ORCA or QE inputs.
        - template_dir (str): Directory containing SLURM templates.
        """
        self.output_dir = output_dir
        self.dft_software = dft_software.lower()
        self.template_dir = template_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_dft_inputs(self, atoms_list):
        """
        Generates DFT input files and corresponding SLURM scripts.

        Parameters:
        - atoms_list (list): List of ASE Atoms objects.

        Returns:
        - input_files (list): List of generated input file paths.
        """
        if self.dft_software == "orca":
            template = os.path.join(self.template_dir, "orca_template.inp")
            inputs = self._write_orca_files(atoms_list, template)
            self.create_slurm_scripts(inputs, "slurm_template_orca.slurm")
        elif self.dft_software == "qe":
            inputs = self._write_qe_files(atoms_list)
            self.create_slurm_scripts(inputs, "slurm_template_qe.slurm")
        else:
            raise ValueError(f"Unsupported DFT software: {self.dft_software}")

        return inputs

    def _write_orca_files(self, atoms_list, template):
        """
        Generates ORCA input files.

        Parameters:
            atoms_list (list): List of ASE Atoms objects.
            template (str): Path to the ORCA template file.

        Returns:
            input_files (list): List of generated ORCA input file paths.
        """
        input_files = []

        logging.info("Writing ORCA input files.")

        for i, atoms in enumerate(atoms_list):
            base_name = f"structure_{i}"
            xyz_file = os.path.join(self.output_dir, f"{base_name}.xyz")
            input_file = os.path.join(self.output_dir, f"{base_name}.inp")

            # Write XYZ file
            write(xyz_file, atoms, format="xyz")

            # Read the ORCA template
            with open(template, "r") as tmpl:
                content = tmpl.read()

            # Append the XYZ file reference
            formatted_content = content + f"\n\n* xyzfile 0 1 {base_name}.xyz\n\n"

            # Write ORCA input file
            with open(input_file, "w") as inp:
                inp.write(formatted_content)

            input_files.append(input_file)
            logging.info(f"Created ORCA input: {input_file}, referencing {xyz_file}")

        return input_files

    def _write_qe_files(self, atoms_list):
        """
        Generates Quantum ESPRESSO input files.

        Parameters:
            atoms_list (list): List of ASE Atoms objects.

        Returns:
            input_files (list): List of generated QE input file paths.
        """
        input_data = {
            "calculation": "scf",
            "prefix": "qe_input",
            "pseudo_dir": "~/QE/pseudo",
            "outdir": "./out/",
            "verbosity": "high",
            "tstress": True,
            "tprnfor": True,
            "ecutrho": 600,
            "ecutwfc": 90,
            "vdw_corr": "mbd",
            "occupations": "smearing",
            "smearing": "cold",
        }
        
        pseudos = {
            "Cl": "Cl.upf",
            "O": "O.upf",
            "F": "F.upf",
            "I": "I.upf",
            "Br": "Br.upf",
            "La": "La.upf",
            "Li": "Li.upf",
            "Zr": "Zr.upf",
            "C": "C.upf",
            "H": "H.upf",
            "Nb": "Nb.upf",
        }

        input_files = []

        logging.info("Writing QE input files.")

        for i, atoms in enumerate(atoms_list):
            filename = os.path.join(self.output_dir, f"qe_input_{i}.in")
            
            write(
                filename=filename,
                images=atoms, 
                format='espresso-in',
                input_data=input_data,
                pseudopotentials=pseudos,
                kspacing=0.05
            )

            input_files.append(filename)
            logging.info(f"Created QE input: {filename}")

        return input_files
    
    def create_slurm_scripts(self, input_files, template_name):
        """
        Generate SLURM job scripts for multiple input files.

        Args:
            input_files (list): List of DFT input file paths.
            template_name (str): Name of the SLURM template file (e.g., "slurm_template_orca.slurm").
        """
        for input_file in input_files:
            base_name = os.path.basename(input_file)  # Extract filename (with extension)
            filename_no_ext, ext = os.path.splitext(base_name)  # Separate name and extension
            slurm_name = filename_no_ext + ".slurm"  # Proper SLURM script name
            self.create_slurm_script(template_name, slurm_name, base_name, filename_no_ext)

    def create_slurm_script(self, template_name, output_name, input_filename, output_filename):
        """
        Generate a single SLURM job script from a template.

        Args:
            template_name (str): Name of the SLURM template file (e.g., "slurm_template_orca.slurm").
            output_name (str): Name of the output SLURM script.
            input_filename (str): Full input filename (e.g., "structure_0.inp").
            output_filename (str): Filename without extension (e.g., "structure_0").
        """
        template_path = os.path.join(self.template_dir, template_name)
        output_path = os.path.join(self.output_dir, output_name)

        # Ensure the template exists
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file '{template_path}' not found.")

        # Read the SLURM template
        with open(template_path, "r") as file:
            slurm_content = file.read()

        # Replace placeholders
        slurm_content = slurm_content.replace("{filename}", input_filename)  # Keep extension for input
        slurm_content = slurm_content.replace("{output_filename}", output_filename)  # Remove extension for output

        # Write the modified content to the output SLURM script
        with open(output_path, "w") as file:
            file.write(slurm_content)

        logging.info(f"Created SLURM script: {output_path}")
   
class DFTOutputParser:
    """
    Parses output files from ORCA and Quantum ESPRESSO (QE).
    Extracts energy, forces, and final structure.
    """

    def __init__(self,output_dir,dft_software):
        """
        Initializes the DFTOutputParser.

        Parameters:
        - output_dir (str): Directory where output files are located.
        - dft_software (str): Either 'orca' or 'qe'.
        """
        self.output_dir = output_dir
        self.dft_software = dft_software.lower()

    def parse_outputs(self):
        """
        Parses all DFT output files in the directory (excluding SLURM and non-.out files).
        Gracefully handles errors in file parsing.

        Returns:
        - results (list of dict): Each dict contains 'filename', 'energy', 'forces', and 'atoms'.
        """
        results = []

        for file in os.listdir(self.output_dir):
            if file.endswith(".out") and not file.startswith("slurm") and not file.endswith(".pw.out"):
                full_path = os.path.join(self.output_dir, file)

                try:
                    if self.dft_software == "orca":
                        result = self._parse_orca_to_ase(full_path)
                    elif self.dft_software == "qe":
                        result = self._parse_qe_output(full_path)
                    else:
                        raise ValueError(f"Unsupported DFT software: {self.dft_software}")

                    if result:
                        results.append(result)

                except Exception as e:
                    logging.warning(f"Failed to parse {file}: {e}")

        return results

    def _parse_orca_to_ase(self,file_path):
        """
        Parses an ORCA output file and returns an ASE Atoms object with:
        - Atomic symbols
        - Positions (Angstroms)
        - Forces (eV/Å) (without negation)
        - Total energy (eV)

        Parameters:
            file_path (str): Path to the ORCA output file.

        Returns:
            Atoms: ASE Atoms object with energy and forces.
        """
        energy = None
        forces = []
        positions = []
        elements = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Extract final total energy (Hartree)
            for line in lines:
                if 'FINAL SINGLE POINT ENERGY' in line:
                    energy = float(line.split()[-1]) * 27.2114  # Convert to eV

            # Extract atomic positions & elements from "CARTESIAN COORDINATES"
            in_positions_section = False
            for i, line in enumerate(lines):
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                    in_positions_section = True
                    continue  # Skip header line

                if in_positions_section:
                    if line.strip() == "":  # End of positions section
                        break
                    data = line.split()
                    if len(data) == 4:
                        elements.append(data[0])  # Store atomic symbol
                        positions.append([float(data[1]), float(data[2]), float(data[3])])  # Store atomic positions

            # Extract Cartesian gradients (forces in Hartree/Bohr)
            in_forces_section = False
            skip_lines = 2  # Number of lines to skip after "CARTESIAN GRADIENT"

            for line in lines:
                if "CARTESIAN GRADIENT" in line:
                    in_forces_section = True
                    skip_lines = 2  # Reset the skip counter
                    continue

                if in_forces_section:
                    if skip_lines > 0:
                        skip_lines -= 1  # Skip the next two lines
                        continue

                    if line.strip() == "":  # Stop when a blank line is encountered
                        break
                    
                    data = line.split()
                    if len(data) >= 6 and data[2] == ":":  # Ensure correct parsing
                        try:
                            forces.append([
                                float(data[3]),  # X force
                                float(data[4]),  # Y force
                                float(data[5])   # Z force
                            ])
                        except ValueError:
                            break  # Stop if we encounter non-numeric data

            # Convert lists to numpy arrays, ensuring they are never None
            positions = np.array(positions) if positions else np.array([])
            forces = np.array(forces) * 51.422 if forces else np.array([])  # Convert Hartree/Bohr to eV/Å
           
            # Validate if element and position lists are the same length
            if len(elements) != len(positions):
                raise ValueError(f"Mismatch in element count ({len(elements)}) and position count ({len(positions)}).")

            # Create ASE Atoms object
            atoms = Atoms(symbols=elements, positions=positions)

            # Attach energy and forces
            if energy is not None:
                atoms.info['dft_energy'] = energy

            # Ensure forces are added even if there’s a mismatch
            if forces.size > 0:
                atoms.arrays['dft_forces'] = forces
            else:
                print(f"Warning: No forces extracted or mismatch in count for {file_path}")

            return atoms

    def _parse_qe_output(self, filepath):
        """
        Parses a single QE output file using ASE's read function.

        Returns:
        - dict: Parsed data with energy, forces, and ASE Atoms object.
        """
        try:
            atoms = read(filepath, format="espresso-out")
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            return atoms

        except Exception as e:
            logging.warning(f"Failed to parse QE output {filepath}: {e}")
            return None
