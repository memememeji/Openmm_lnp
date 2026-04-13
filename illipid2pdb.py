import pandas as pd
import numpy as np
from openff.toolkit import Molecule
from openmm.app import ForceField
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem

def smiles2pdb(smiles,path):
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    mol.assign_partial_charges("gasteiger")
    rdmol = mol.to_rdkit()
    Chem.MolToPDBFile(rdmol, path)
    return None

def illipid2pdb(index,smiles):
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    mol.assign_partial_charges("gasteiger")
    rdmol = mol.to_rdkit()
    Chem.MolToPDBFile(rdmol,f"pdb_folder/{index}.pdb")
    return None


def illipidcsv2pdb(path):
    illipid_file = pd.read_csv(path)
    # 仅提取其中label,combined_mol_SMILES两列
    illipid_file = illipid_file[['index', 'smiles']]
    for i in range(len(illipid_file)):
        index = illipid_file.iloc[i]['index']
        smiles = illipid_file.iloc[i]['smiles']
        illipid2pdb(index, smiles)
    return None

def main():
    illipidcsv2pdb("data/Agile_finetune.csv")
    # smiles2pdb("CCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCCNCCOCCOCCOCCOCCOCCOCCOCCOCCOC)OC(=O)CCCCCCCCCCCCCC", "data/PEG_smiles.pdb")
    return None

if __name__ == '__main__':
    main()