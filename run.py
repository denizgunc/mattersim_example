import torch
from loguru import logger
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator


filename = 'MnBiEuCuO4.xyz'

atoms = read(filename)
print(atoms)
atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")

logger.info('Original Structure:')
logger.info(f"Energy (eV)                 = {atoms.get_potential_energy()}")
logger.info(f"Energy per atom (eV/atom)   = {atoms.get_potential_energy()/len(atoms)}")
logger.info(f"Forces of first atom (eV/A) = {atoms.get_forces()[0]}")
logger.info(f"Stress[0][0] (eV/A^3)       = {atoms.get_stress(voigt=False)[0][0]}")
logger.info(f"Stress[0][0] (GPa)          = {atoms.get_stress(voigt=False)[0][0] / GPa}")

ase_filter = FrechetCellFilter(atoms)
optimizer = BFGS(ase_filter, trajectory='relax.traj')
optimizer.run(fmax=0.01)

logger.info('Structure after relaxation:')
logger.info(f"Energy (eV)                 = {atoms.get_potential_energy()}")
logger.info(f"Energy per atom (eV/atom)   = {atoms.get_potential_energy()/len(atoms)}")
logger.info(f"Forces of first atom (eV/A) = {atoms.get_forces()[0]}")
logger.info(f"Stress[0][0] (eV/A^3)       = {atoms.get_stress(voigt=False)[0][0]}")
logger.info(f"Stress[0][0] (GPa)          = {atoms.get_stress(voigt=False)[0][0] / GPa}")

from pathlib import Path
def append_to_filename_pathlib(filename, text):
    path = Path(filename)
    return str(path.parent / f"{path.stem} {text}{path.suffix}")
relaxed_filename = append_to_filename_pathlib(filename, 'mattersimrelaxed')
logger.info(f'writing output to: {relaxed_filename}')
write(relaxed_filename, atoms)
write(f'{relaxed_filename}_0x0y0z.png', atoms, rotation='0x,0y,0z')
write(f'{relaxed_filename}_90x0y0z.png', atoms, rotation='90x,0y,0z')
write(f'{relaxed_filename}_0x90y0z.png', atoms, rotation='0x,90y,0z')
write(f'{relaxed_filename}_0x0y90z.png', atoms, rotation='0x,0y,90z')

