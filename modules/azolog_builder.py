# modules/azolog_builder.py

from rdkit import Chem
from rdkit.Chem import AllChem
import copy

def generate_trans_cis_pair(parent_smiles):
    """
    Takes a natural phytochemical SMILES and generates hypothetical
    trans and cis isomers by inserting an azo (N=N) bridge.
    
    This is a conceptual simulation for educational demonstration.
    In a production system, we would use precise reaction SMARTS.
    """
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return None, None, None, None
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Create "trans" (elongated) - original geometry
    trans_mol = copy.deepcopy(mol)
    trans_smiles = Chem.MolToSmiles(trans_mol, isomericSmiles=True)
    
    # Create "cis" (bent) - modified geometry
    # For educational simulation, we rotate a rotatable bond to create a bent conformer
    cis_mol = copy.deepcopy(mol)
    rotatable_bonds = cis_mol.GetSubstructMatches(
        Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]')
    )
    
    if rotatable_bonds:
        # Get the first rotatable bond
        bond = cis_mol.GetBondBetweenAtoms(
            rotatable_bonds[0][0], rotatable_bonds[0][1]
        )
        if bond:
            # Try to rotate to create a different conformer
            AllChem.SetDihedralDeg(
                cis_mol.GetConformer(),
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBeginAtom().GetNeighbors()[0].GetIdx() if bond.GetBeginAtom().GetDegree() > 1 else 0,
                bond.GetEndAtom().GetNeighbors()[0].GetIdx() if bond.GetEndAtom().GetDegree() > 1 else 0,
                120.0  # Rotate 120 degrees to create "bent" isomer
            )
    
    cis_smiles = Chem.MolToSmiles(cis_mol, isomericSmiles=True)
    
    return trans_smiles, cis_smiles, trans_mol, cis_mol

def get_molecular_properties(smiles):
    """Calculate basic molecular properties."""
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logP": round(Descriptors.MolLogP(mol), 2),
        "h_bond_donors": Descriptors.NumHDonors(mol),
        "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
    }
