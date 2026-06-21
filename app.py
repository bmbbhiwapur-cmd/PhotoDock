# Add to modules/utils.py

def check_lipinski(smiles):
    """Check Lipinski Rule of 5 compliance."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'mw_pass': False, 'logp_pass': False, 'hbd_pass': False, 
                'hba_pass': False, 'rotb_pass': False}
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    
    return {
        'mw': mw, 'mw_pass': mw <= 500,
        'logp': logp, 'logp_pass': logp <= 5,
        'hbd': hbd, 'hbd_pass': hbd <= 5,
        'hba': hba, 'hba_pass': hba <= 10,
        'rotb': rotb, 'rotb_pass': rotb <= 10,
    }

def get_administration_route(smiles):
    """Determine recommended administration route based on properties."""
    lip = check_lipinski(smiles)
    total_pass = sum([lip['mw_pass'], lip['logp_pass'], lip['hbd_pass'], 
                      lip['hba_pass'], lip['rotb_pass']])
    
    if total_pass >= 4 and lip['logp'] < 5:
        return "Oral or Topical (Lipinski compliant — suitable for both internal and external use per IKS principles)"
    elif lip['logp'] < 3:
        return "Topical / External only (per Sūrya-saṃyoga-cikitsā tradition — direct sun exposure needed)"
    else:
        return "Topical with photoactivation (matches traditional IKS external application with sunlight)"
