import time
import streamlit as st
import subprocess
import os
import shutil
import urllib.request
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# --- CLOUD CONTEXT ENGINE MANAGEMENT ---
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")

ensure_linux_vina_exists()

# --- PHASE 3: SURYA-SANYOG AZOLOGIZATION ENGINE ---
class Azologizer:
    def __init__(self):
        self.reaction_smarts = '[c:1]-[CH1:2]=[CH1:3]-[c:4]>>[c:1]-[N:2]=[N:3]-[c:4]'
        self.rxn = AllChem.ReactionFromSmarts(self.reaction_smarts)

    def mutate_ligand(self, native_smiles):
        native_mol = Chem.MolFromSmiles(native_smiles)
        if not native_mol: return None
        products = self.rxn.RunReactants((native_mol,))
        if not products: return None
        azo_mol = products[0][0]
        Chem.SanitizeMol(azo_mol)
        return Chem.MolToSmiles(azo_mol)

    def get_2d_isomer_image(self, azo_smiles, isomer_type="trans"):
        mol = Chem.MolFromSmiles(azo_smiles)
        azo_bond = None
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                if bond.GetBeginAtom().GetAtomicNum() == 7 and bond.GetEndAtom().GetAtomicNum() == 7:
                    azo_bond = bond
                    break
        if azo_bond:
            if isomer_type.lower() == "trans": azo_bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
            elif isomer_type.lower() == "cis": azo_bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            try:
                b_idx = [a.GetIdx() for a in azo_bond.GetBeginAtom().GetNeighbors() if a.GetIdx() != azo_bond.GetEndAtomIdx()][0]
                e_idx = [a.GetIdx() for a in azo_bond.GetEndAtom().GetNeighbors() if a.GetIdx() != azo_bond.GetBeginAtomIdx()][0]
                azo_bond.SetStereoAtoms(b_idx, e_idx)
            except IndexError: pass
        AllChem.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=(350, 300), fitImage=True)

    def generate_3d_isomer_pdbqt(self, azo_smiles, isomer_type="trans", output_filename="isomer.pdbqt"):
        mol = Chem.MolFromSmiles(azo_smiles)
        azo_bond = None
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                if bond.GetBeginAtom().GetAtomicNum() == 7 and bond.GetEndAtom().GetAtomicNum() == 7:
                    azo_bond = bond
                    break
        if not azo_bond: return False, "No N=N bond found."
        if isomer_type.lower() == "trans": azo_bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
        elif isomer_type.lower() == "cis": azo_bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
        try:
            begin_neighbor = [a.GetIdx() for a in azo_bond.GetBeginAtom().GetNeighbors() if a.GetIdx() != azo_bond.GetEndAtomIdx()][0]
            end_neighbor = [a.GetIdx() for a in azo_bond.GetEndAtom().GetNeighbors() if a.GetIdx() != azo_bond.GetBeginAtomIdx()][0]
            azo_bond.SetStereoAtoms(begin_neighbor, end_neighbor)
        except IndexError: pass

        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.enforceChirality = True
        if AllChem.EmbedMolecule(mol_3d, params) != 0: return False, "Failed to embed 3D coordinates."
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        temp_pdb = f"temp_{isomer_type}.pdb"
        Chem.MolToPDBFile(mol_3d, temp_pdb)
        ok, msg = convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return ok, output_filename

# --- BIOINFORMATICS PARSERS & CONVERTERS ---
def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound Name", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical Derivative")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception: pass 
    return metadata

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {"name": "Unknown Protein", "title": "Uploaded Protein Structure Matrix", "id": pdb_id.upper() if pdb_id and pdb_id != "Uploaded File" else "Unknown", "class": "Unknown Classification", "organism": "Unknown", "system": "Unknown Expression System", "method": "X-RAY DIFFRACTION", "res": "N/A"}
    if not os.path.exists(file_path): return meta
    with open(file_path, "r") as f:
        title_parts = []
        for line in f:
            if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
            elif line.startswith("HEADER"): 
                meta["class"] = line[10:50].strip().title()
                if len(line) >= 66:
                    possible_id = line[62:66].strip()
                    if len(possible_id) == 4: meta["id"] = possible_id.upper()
            elif line.startswith("COMPND"):
                if "MOLECULE:" in line:
                    mol_name = line.split("MOLECULE:")[1].split(";")[0].strip()
                    if meta["name"] == "Unknown Protein": meta["name"] = mol_name.title()
            elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
            elif "EXPRESSION_SYSTEM" in line: meta["system"] = line.split(":")[-1].replace(";","").strip()
            elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
            elif "RESOLUTION." in line and "ANGSTROMS." in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match: meta["res"] = f"{match.group(1)} Å"
    if title_parts: meta["title"] = " ".join(title_parts).title()
    if meta["name"] == "Unknown Protein" and meta["title"] != "Uploaded Protein Structure Matrix": meta["name"] = meta["title"]
    return meta

def discover_and_list_all_heteroatoms(file_path):
    hetero_counts = {}
    if not os.path.exists(file_path): return hetero_counts
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name in ["HOH", "WAT", "DOD"]: continue
                hetero_counts[res_name] = hetero_counts.get(res_name, 0) + 1
    return hetero_counts

def parse_bound_ligands(file_path):
    ligands = {}
    if not os.path.exists(file_path): return ligands
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip() if line[21].strip() else "A"
                try: res_seq = int(line[22:26].strip())
                except ValueError: continue
                if res_name in ["HOH", "WAT", "DOD"]: continue
                key = f"{res_name}-{chain_id}-{res_seq}"
                try:
                    x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                except ValueError: continue
                if key not in ligands: ligands[key] = {"res": res_name, "chain": chain_id, "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))
    processed_ligands = []
    for key, info in ligands.items():
        pts = info["coords"]
        n_atoms = len(pts)
        if n_atoms < 4: continue
        cx, cy, cz = sum([p[0] for p in pts])/n_atoms, sum([p[1] for p in pts])/n_atoms, sum([p[2] for p in pts])/n_atoms
        bx = max([p[0] for p in pts]) - min([p[0] for p in pts]) + 10.0
        by = max([p[1] for p in pts]) - min([p[1] for p in pts]) + 10.0
        bz = max([p[2] for p in pts]) - min([p[2] for p in pts]) + 10.0
        processed_ligands.append({"ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms, "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2), "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)})
    return processed_ligands

def identify_protein_cavities(pdbqt_file, max_pockets=5):
    coords = []
    if not os.path.exists(pdbqt_file): return []
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if len(coords) < 10: return []
    arr = np.array(coords)
    min_bound = np.min(arr, axis=0)
    max_bound = np.max(arr, axis=0)
    step = (max_bound - min_bound) / 4.0
    pockets = []
    idx = 1
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                pt = min_bound + np.array([i*step[0], j*step[1], k*step[2]])
                dists = np.linalg.norm(arr - pt, axis=1)
                score = np.sum((dists > 3.0) & (dists < 12.0))
                core_clash = np.sum(dists <= 3.0)
                if core_clash < 20 and score > 20:
                    pockets.append({"Pocket_ID": f"Cavity {idx}", "cx": round(pt[0], 2), "cy": round(pt[1], 2), "cz": round(pt[2], 2), "bx": 20.0, "by": 20.0, "bz": 20.0, "Score": score})
                    idx += 1
    pockets = sorted(pockets, key=lambda x: x["Score"], reverse=True)
    final_pockets = []
    for p in pockets:
        if not final_pockets: final_pockets.append(p)
        else:
            is_unique = True
            for fp in final_pockets:
                dist = np.linalg.norm(np.array([p["cx"], p["cy"], p["cz"]]) - np.array([fp["cx"], fp["cy"], fp["cz"]]))
                if dist < 6.0: is_unique = False; break
            if is_unique: final_pockets.append(p)
        if len(final_pockets) >= max_pockets: break
    if not final_pockets:
        center = np.mean(arr, axis=0)
        dims = max_bound - min_bound
        final_pockets.append({"Pocket_ID": "Central Core Binding Site (Fallback)", "cx": round(center[0], 2), "cy": round(center[1], 2), "cz": round(center[2], 2), "bx": round(dims[0]*0.5, 2) + 5, "by": round(dims[1]*0.5, 2) + 5, "bz": round(dims[2]*0.5, 2) + 5, "Score": 100})
    return final_pockets

def compute_protein_centroid(pdbqt_file):
    coords = []
    if not os.path.exists(pdbqt_file): return 0.0, 0.0, 0.0, 20.0, 20.0, 20.0
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if not coords: return 0.0, 0.0, 0.0, 20.0, 20.0, 20.0
    arr = np.array(coords)
    center = np.mean(arr, axis=0)
    dims = np.max(arr, axis=0) - np.min(arr, axis=0) + 10.0 
    return center[0], center[1], center[2], min(60.0, dims[0]), min(60.0, dims[1]), min(60.0, dims[2])

def parse_pdbqt_coordinates(pdbqt_string):
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                element = line[76:78].strip().upper()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    with open(receptor_file, "r") as f: receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]: b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C" and any(aro in r_at["res"] for aro in ["PHE", "TYR", "TRP"])): b_type = "pi-Stacking / Hydrophobic"
                else: b_type = "van der Waals Contact"
                seen.add(res_id)
                interactions.append({"Residue Contact": res_id, "Interaction Type": b_type, "Distance (Å)": round(dist, 2), "r_coord": r_at["coord"].tolist(), "l_coord": l_at["coord"].tolist()})
    return interactions

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception: return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False, allowed_heteroatoms=None):
    if allowed_heteroatoms is None: allowed_heteroatoms = []
    autodock_type_map = {"H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg", "FE": "Fe", "CA": "Ca"}
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception: torsions = 4
    temp_out = f"temp_safe_write_{output_pdbqt}"
    try:
        atom_count = 0
        with open(input_pdb, "r") as pdb, open(temp_out, "w") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith("ATOM") or (line.startswith("HETATM") and not is_ligand) or (line.startswith("HETATM") and is_ligand):
                    record_type = line[:6].strip()
                    res_name = line[17:20].strip()
                    if record_type == "HETATM" and not is_ligand and res_name not in allowed_heteroatoms: continue
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    chain_id = line[21].strip() if line[21].strip() else "A"
                    try: res_seq = int(line[22:26].strip())
                    except ValueError: res_seq = 1
                    try: x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    except ValueError: continue
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"
                    prefix = "ATOM  "
                    pdbqt.write(f"{prefix}{atom_id:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vina_type:<2}\n")
                    atom_count += 1
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        shutil.move(temp_out, output_pdbqt)
        return atom_count > 0, output_pdbqt
    except Exception as e:
        if os.path.exists(temp_out): os.remove(temp_out)
        return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        ok, _ = convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return ok, output_filename
    except Exception as e: return False, str(e)

def execute_uff_complex_minimization(protein_path, ligand_pose_str, progress_ui=None):
    try:
        protein_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
        ligand_mol = Chem.MolFromPDBBlock(ligand_pose_str, sanitize=False, removeHs=False)
        if not protein_mol or not ligand_mol: return "N/A", "N/A", "N/A"
        total_atoms = protein_mol.GetNumAtoms() + ligand_mol.GetNumAtoms()
        MAX_SAFE_ATOMS = 4000 
        if total_atoms > MAX_SAFE_ATOMS:
            if progress_ui: progress_ui.warning(f"⚠️ UFF Skipped: Complex too massive ({total_atoms} atoms).")
            time.sleep(1.5) 
            return "Bypassed", "Bypassed", "N/A"
        combined_complex = Chem.CombineMols(protein_mol, ligand_mol)
        try: Chem.SanitizeMol(combined_complex, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception: pass
        uff_field = AllChem.UFFGetMoleculeForceField(combined_complex)
        if not uff_field: return "N/A", "N/A", "N/A"
        pre_energy = uff_field.CalcEnergy()
        max_iter = 150
        chunk_size = 15
        if progress_ui: prog_bar = progress_ui.progress(0, text="⏳ Initializing UFF Force Field Physics Matrix...")
        res = 1
        for i in range(0, max_iter, chunk_size):
            res = uff_field.Minimize(maxIts=chunk_size, forceTol=1e-3)
            pct = min(100, int(((i + chunk_size) / max_iter) * 100))
            if progress_ui: prog_bar.progress(pct, text=f"🧬 Relaxing Complex Sterics... ({pct}% complete)")
            time.sleep(0.01) 
            if res == 0:
                if progress_ui: prog_bar.progress(100, text="✨ Steric Relaxation Converged Perfectly!")
                break
        if res != 0 and progress_ui: prog_bar.progress(100, text="✨ Steric Relaxation Completed.")
        post_energy = uff_field.CalcEnergy()
        delta_energy = post_energy - pre_energy
        time.sleep(0.4)
        return f"{pre_energy:.2f}", f"{post_energy:.2f}", f"{delta_energy:.2f}"
    except MemoryError:
        if progress_ui: progress_ui.error("🚨 Out of Memory Error during UFF calculation. Bypassing.")
        return "OOM Crash", "OOM Crash", "N/A"
    except Exception: return "N/A", "N/A", "N/A"

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def get_top_affinity_from_pdbqt(file_path):
    if not os.path.exists(file_path): return 0.0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                return float(line.split()[3])
    return 0.0

def parse_vina_output_with_residues(stdout_text):
    data = []
    poses_dict = split_docking_poses("docking_poses.pdbqt")
    if not stdout_text: return pd.DataFrame(data)
    for line in stdout_text.split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                mode_idx = int(parts[0])
                aff = float(parts[1])
                rmsd_lb = float(parts[2])
                rmsd_ub = float(parts[3])
                res_string, bond_types = "N/A", "N/A"
                if mode_idx in poses_dict:
                    ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mode_idx])
                    if ints:
                        res_string = ", ".join(list(set([i["Residue Contact"] for i in ints])))
                        bond_types = ", ".join(list(set([i["Interaction Type"] for i in ints])))
                data.append({"Binding Mode": mode_idx, "Affinity (kcal/mol)": aff, "RMSD l.b.": rmsd_lb, "RMSD u.b.": rmsd_ub, "Interacting Residues": res_string, "Contact Bond Types": bond_types})
            except ValueError: continue
    return pd.DataFrame(data)

def build_styled_html_table(df):
    html = '<table class="data-table" border="1" cellpadding="5" style="border-collapse: collapse; width:100%; text-align:left;"><thead><tr>'
    for col in df.columns: html += f'<th style="background-color:#f2f2f2;">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            val = row[col]
            if col == 'Affinity (kcal/mol)':
                try:
                    if float(val) > 0: html += f'<td style="color: #c62828; font-weight: bold;">{val}</td>'
                    else: html += f'<td style="color: #1b5e20;">{val}</td>'
                except: html += f'<td>{val}</td>'
            else: html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

# --- VISUALIZATION CONSTRUCTS ---
def render_isomer_comparison(trans_data, cis_data):
    html_content = f"""
    <div style="display: flex; width: 100%; justify-content: space-between; gap: 10px;">
        <div id="trans_view" style="width: 50%; height: 350px; border:2px solid #2e7d32; border-radius:8px; position:relative; background:#ffffff;">
            <div style="position:absolute; top:8px; left:8px; z-index:10; font-weight:bold; color:#2e7d32; background:rgba(255,255,255,0.9); padding:4px 8px; border-radius:4px; border:1px solid #2e7d32;">Trans Isomer (Dark State)</div>
        </div>
        <div id="cis_view" style="width: 50%; height: 350px; border:2px solid #c62828; border-radius:8px; position:relative; background:#ffffff;">
            <div style="position:absolute; top:8px; left:8px; z-index:10; font-weight:bold; color:#c62828; background:rgba(255,255,255,0.9); padding:4px 8px; border-radius:4px; border:1px solid #c62828;">Cis Isomer (Light State)</div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let v_trans = $3Dmol.createViewer(document.getElementById('trans_view'), {{backgroundColor: '#f8fbf8'}});
        v_trans.addModel(`{trans_data}`, 'pdb');
        v_trans.setStyle({{}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.2}}, sphere: {{radius:0.4}}}});
        v_trans.zoomTo(); v_trans.render();

        let v_cis = $3Dmol.createViewer(document.getElementById('cis_view'), {{backgroundColor: '#fff8f8'}});
        v_cis.addModel(`{cis_data}`, 'pdb');
        v_cis.setStyle({{}}, {{stick: {{colorscheme: 'orangeCarbon', radius: 0.2}}, sphere: {{radius:0.4}}}});
        v_cis.zoomTo(); v_cis.render();
    </script>
    """
    components.html(html_content, height=370)

def render_dual_docking_viewport(protein_data, trans_pose, cis_pose, int_t=[], int_c=[]):
    
    int_js_t = ""
    for interact in int_t:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_js_t += f"""
        v_t.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        v_t.addLabel("{interact['Residue Contact']} ({interact['Distance (Å)']}A)", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """
        
    int_js_c = ""
    for interact in int_c:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_js_c += f"""
        v_c.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        v_c.addLabel("{interact['Residue Contact']} ({interact['Distance (Å)']}A)", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """

    html_content = f"""
    <div style="display: flex; width: 100%; justify-content: space-between; gap: 10px;">
        <div id="dock_trans" style="width: 50%; height: 450px; border:2px solid #2e7d32; border-radius:8px; position:relative; background:#ffffff;">
            <div style="position:absolute; top:8px; left:8px; z-index:10; font-weight:bold; color:#2e7d32; background:rgba(255,255,255,0.9); padding:4px 8px; border-radius:4px; border:1px solid #2e7d32;">Active: Trans Pose in Cavity</div>
        </div>
        <div id="dock_cis" style="width: 50%; height: 450px; border:2px solid #c62828; border-radius:8px; position:relative; background:#ffffff;">
            <div style="position:absolute; top:8px; left:8px; z-index:10; font-weight:bold; color:#c62828; background:rgba(255,255,255,0.9); padding:4px 8px; border-radius:4px; border:1px solid #c62828;">Inactive: Cis Pose in Cavity</div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let v_t = $3Dmol.createViewer(document.getElementById('dock_trans'), {{backgroundColor: '#ffffff'}});
        v_t.addModel(`{protein_data}`, 'pdb');
        v_t.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
        v_t.addModel(`{trans_pose}`, 'pdb');
        v_t.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        {int_js_t}
        v_t.zoomTo(); v_t.render();

        let v_c = $3Dmol.createViewer(document.getElementById('dock_cis'), {{backgroundColor: '#ffffff'}});
        v_c.addModel(`{protein_data}`, 'pdb');
        v_c.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
        v_c.addModel(`{cis_pose}`, 'pdb');
        v_c.setStyle({{model: 1}}, {{stick: {{colorscheme: 'orangeCarbon', radius: 0.28}}}});
        {int_js_c}
        v_c.zoomTo(); v_c.render();
    </script>
    """
    components.html(html_content, height=470)

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[]):
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc = interact["r_coord"]
        lc = interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer.addLabel("{interact['Residue Contact']} ({interact['Distance (Å)']}A)", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:11}});
        """
    html_content = f"""
    <div id="wrapper_div" style="position:relative; width:100%;">
        <button onclick="toggleFullScreen()" style="position:absolute; top:12px; right:12px; z-index:9999; padding:6px 12px; background:#007bff; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-family:sans-serif; box-shadow:0 2px 4px rgba(0,0,0,0.15);">🖥 Fullscreen View</button>
        <div id="container" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{ viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
            }} else if ('{mode}' === 'spacefill') {{ viewer.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}});
            }} else {{ viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}}); }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer.zoomTo(); viewer.render();
        function toggleFullScreen() {{
            let elem = document.getElementById("wrapper_div");
            if (!document.fullscreenElement) {{ elem.requestFullscreen(); document.getElementById("container").style.height = "90vh"; }}
            else {{ document.exitFullscreen(); document.getElementById("container").style.height = "480px"; }}
        }}
        document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) document.getElementById("container").style.height = "480px"; }});
    </script>
    """
    components.html(html_content, height=510)


# ==========================================
# APPLICATION DASHBOARD WORKSPACE (UI)
# ==========================================

st.set_page_config(page_title="PhotoDock (सूर्य-संयोग-चिकित्सा)", layout="wide")
st.title("🔬 PhotoDock (सूर्य-संयोग-चिकित्सा) - Precision Photopharmacology")
st.markdown("**From Ancient Sun Therapy to Red-Light Precision Antibiotics.** Developed by: Tech Logic Core Systems (TLCS) | Lead Architect: Dr. Sarang S. Dhote.")

# Initialize states safely
if "protein_name" not in st.session_state: st.session_state.protein_name = "Unknown Protein"
if "pdb_id_display" not in st.session_state: st.session_state.pdb_id_display = "Custom"
if "cx" not in st.session_state: st.session_state.cx = 0.0
if "cy" not in st.session_state: st.session_state.cy = 0.0
if "cz" not in st.session_state: st.session_state.cz = 0.0
if "sx" not in st.session_state: st.session_state.sx = 20
if "sy" not in st.session_state: st.session_state.sy = 20
if "sz" not in st.session_state: st.session_state.sz = 20
if "target_ready" not in st.session_state: st.session_state.target_ready = False
if "ligand_ready" not in st.session_state: st.session_state.ligand_ready = False
if "local_target_path" not in st.session_state: st.session_state.local_target_path = None
if "docking_results_raw" not in st.session_state: st.session_state.docking_results_raw = None
if "serialized_ligand_block" not in st.session_state: st.session_state.serialized_ligand_block = None
if "ligand_summary_text" not in st.session_state: st.session_state.ligand_summary_text = ""
if "smiles_cache" not in st.session_state: st.session_state.smiles_cache = ""
if "mutated_azo_smiles" not in st.session_state: st.session_state.mutated_azo_smiles = ""
if "selected_native_ligand" not in st.session_state: st.session_state.selected_native_ligand = "None (Manual / Blind Docking)"
if "detected_pockets" not in st.session_state: st.session_state.detected_pockets = []
if "active_retained_ions" not in st.session_state: st.session_state.active_retained_ions = "None"
if "uff_cache" not in st.session_state: st.session_state.uff_cache = {}
if "last_uploaded_protein" not in st.session_state: st.session_state.last_uploaded_protein = ""
if "last_uploaded_ligand" not in st.session_state: st.session_state.last_uploaded_ligand = "" 
if "comparative_run_complete" not in st.session_state: st.session_state.comparative_run_complete = False

if st.button("🔄 Reset Entire Environment", type="secondary"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "ligand_native.pdbqt", "docking_poses.pdbqt", "docking_native.pdbqt", "docking_trans.pdbqt", "docking_cis.pdbqt", "temp_lig_state.pdb", "ligand_trans.pdbqt", "ligand_cis.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.rerun()

col_params, col_visual = st.columns([1, 1])

with col_params:
    st.header("1. Target Protein Setup")
    
    current_p_name = st.text_input("Protein Name", placeholder="Hint: Type protein name here...", value=st.session_state.protein_name)
    current_p_id = st.text_input("PDB ID / Code", placeholder="Hint: Type PDB ID here...", value=st.session_state.pdb_id_display)
    
    if current_p_name != st.session_state.protein_name: st.session_state.protein_name = current_p_name
    if current_p_id != st.session_state.pdb_id_display: st.session_state.pdb_id_display = current_p_id

    cancer_receptors = {
        "EGFR Kinase Domain (Lung/Breast Cancer)": "1M17",
        "Estrogen Receptor Alpha (Breast Cancer)": "3ERT",
        "BRAF V600E Mutant (Melanoma)": "4HJO",
        "GLUT4 Glucose Transporter (Diabetes)": "7WSM"
    }

    protein_source = st.radio("Choose Protein Input Method:", ["Select Curated Target Receptor", "Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    pdb_id_input = ""
    if protein_source == "Select Curated Target Receptor":
        selected_receptor = st.selectbox("Select Target Receptor:", list(cancer_receptors.keys()))
        pdb_id_input = cancer_receptors[selected_receptor]
        st.info(f"Targeting PDB ID: `{pdb_id_input}`")
    elif protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()

    if protein_source in ["Select Curated Target Receptor", "Type 4-Letter PDB ID"]:
        if st.button("📥 Load Target Structure"):
            if pdb_id_input:
                success, path = fetch_pdb_from_rcsb(pdb_id_input)
                if success:
                    st.session_state.local_target_path = path
                    meta = extract_pdb_metadata(path, pdb_id_input.upper())
                    st.session_state.pdb_id_display = meta["id"]
                    st.session_state.protein_name = meta["name"]
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt", allowed_heteroatoms=[])
                    st.session_state.target_ready = conv_ok
                    st.session_state.active_retained_ions = "None (Fully Stripped)"
                    st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
                    st.rerun()
                else: st.error(path)
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            if st.session_state.last_uploaded_protein != uploaded_file.name:
                path = f"uploaded_{uploaded_file.name}"
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.local_target_path = path
                meta = extract_pdb_metadata(path, "Uploaded File")
                st.session_state.pdb_id_display = meta["id"]
                st.session_state.protein_name = meta["name"]
                
                if uploaded_file.name.lower().endswith(".pdb"):
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt", allowed_heteroatoms=[])
                    st.session_state.target_ready = conv_ok
                    st.session_state.active_retained_ions = "None (Fully Stripped)"
                else:
                    os.replace(path, "protein.pdbqt")
                    st.session_state.target_ready = True
                    st.session_state.local_target_path = "protein.pdbqt"
                    st.session_state.active_retained_ions = "Pre-compiled PDBQT (Ions Unknown)"
                
                st.session_state.last_uploaded_protein = uploaded_file.name
                st.rerun()

    if st.session_state.target_ready and st.session_state.local_target_path:
        discovered_het = discover_and_list_all_heteroatoms(st.session_state.local_target_path)
        if discovered_het:
            st.subheader("🧬 Catalytic Cofactors & Heteroatom Filter")
            selected_hets = []
            cols_het = st.columns(min(len(discovered_het), 4))
            for idx, (het_id, count) in enumerate(discovered_het.items()):
                with cols_het[idx % 4]:
                    if st.checkbox(f"Keep {het_id} ({count})", value=False, key=f"keep_het_{het_id}"):
                        selected_hets.append(het_id)
            if st.button("🛠 Rebuild Clean Receptor Structure"):
                ok, err = convert_pdb_to_pdbqt(st.session_state.local_target_path, "protein.pdbqt", is_ligand=False, allowed_heteroatoms=selected_hets)
                if ok:
                    st.session_state.active_retained_ions = ", ".join(selected_hets) if selected_hets else "None (Fully Stripped)"
                    st.success(f"Receptor rebuilt! Retained: {st.session_state.active_retained_ions}")
                    st.session_state.detected_pockets = [] 
                else: st.error(f"Receptor optimization failure: {err}")

    st.write("---")
    st.header("2. Phase 1: Small Molecule Phytochemical Setup")
    iks_library = {
        "Pterostilbene (Antidiabetic / Anticancer)": "COc1cc(C=Cc2ccc(O)cc2)cc(OC)c1",
        "Resveratrol (Antidiabetic / Anticancer)": "Oc1cc(O)cc(C=Cc2ccc(O)cc2)c1",
        "Combretastatin A-4 (Anticancer)": "COc1cc(C=Cc2ccc(O)c(OC)c2)cc(OC)c1OC",
    }
    selected_phytochemical = st.selectbox("Select Native Trans-Stilbene Scaffold:", list(iks_library.keys()))
    smiles_input_val = iks_library[selected_phytochemical]

    if st.button("📥 Load Ligand Structure"):
        pub_data = fetch_ligand_data_from_pubchem(smiles_input_val)
        ok, _ = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
        if ok:
            shutil.copy("ligand.pdbqt", "ligand_native.pdbqt")
            st.session_state.ligand_ready = True
            st.session_state.smiles_cache = smiles_input_val
            st.success(f"Loaded: {pub_data['name']}")

    if st.session_state.ligand_ready and st.session_state.smiles_cache:
        st.write("---")
        st.subheader("💡 Phase 3: Surya-Sanyog Azologization")
        
        if st.button("🧬 Run In Silico Azologization", type="primary"):
            with st.spinner("Generating 3D Isomers..."):
                engine = Azologizer()
                mutated_smiles = engine.mutate_ligand(st.session_state.smiles_cache)
                if mutated_smiles:
                    st.session_state.mutated_azo_smiles = mutated_smiles
                    st.success(f"Mutation Successful! New Azo-Scaffold: `{mutated_smiles}`")
                    trans_ok, trans_file = engine.generate_3d_isomer_pdbqt(mutated_smiles, "trans", "ligand_trans.pdbqt")
                    cis_ok, cis_file = engine.generate_3d_isomer_pdbqt(mutated_smiles, "cis", "ligand_cis.pdbqt")
                    
                    if trans_ok and cis_ok:
                        st.session_state.has_isomers = True
                        st.info("Successfully folded Trans and Cis 3D geometries.")
                    else: st.error("Failed to map constrained 3D coordinates.")
                else: st.warning("No valid C=C bridge found.")

    if st.session_state.get("has_isomers", False):
        st.write("---")
        st.header("3. Grid Box Mechanics")
        if st.button("🌐 Enable Blind Docking (Full Surface)", use_container_width=True):
            cx, cy, cz, sx, sy, sz = compute_protein_centroid("protein.pdbqt")
            st.session_state.cx, st.session_state.cy, st.session_state.cz = cx, cy, cz
            st.session_state.sx, st.session_state.sy, st.session_state.sz = sx, sy, sz
            st.rerun()

        c_col1, c_col2, c_col3 = st.columns(3)
        grid_cx = c_col1.number_input("Center X", value=float(st.session_state.cx), step=1.0)
        grid_cy = c_col2.number_input("Center Y", value=float(st.session_state.cy), step=1.0)
        grid_cz = c_col3.number_input("Center Z", value=float(st.session_state.cz), step=1.0)
        
        grid_sx = c_col1.number_input("Size X", value=float(st.session_state.sx), step=1.0)
        grid_sy = c_col2.number_input("Size Y", value=float(st.session_state.sy), step=1.0)
        grid_sz = c_col3.number_input("Size Z", value=float(st.session_state.sz), step=1.0)

        exhaustiveness = st.slider("Search Exhaustiveness", 4, 32, 8)
        
        can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)

        st.write("---")
        st.header("🚀 4. Execute Comparative Photo-Docking")
        st.markdown("This will run AutoDock Vina sequentially on the **Native** (Original), **Trans** (Dark State) and **Cis** (Light State) isomers in the exact same receptor cavity to construct a comparative matrix.")
        
        if st.button("⚡ Run Full Comparative Docking Sequence", type="primary", disabled=not can_dock):
            progress_bar = st.progress(0, text="Docking Native Ligand...")
            
            if os.path.exists("ligand_native.pdbqt"):
                cmd_native = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_native.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_native.pdbqt"]
                subprocess.run(cmd_native, capture_output=True, text=True)
                
            progress_bar.progress(33, text="Docking Trans Isomer (Dark State)...")
            
            cmd_trans = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_trans.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_trans.pdbqt"]
            t_proc = subprocess.run(cmd_trans, capture_output=True, text=True)
            st.session_state.raw_trans_log = t_proc.stdout
            
            progress_bar.progress(66, text="Docking Cis Isomer (Light State)...")
            
            cmd_cis = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_cis.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_cis.pdbqt"]
            c_proc = subprocess.run(cmd_cis, capture_output=True, text=True)
            st.session_state.raw_cis_log = c_proc.stdout
            
            progress_bar.progress(100, text="Comparative Docking Complete!")
            st.session_state.comparative_run_complete = True
            time.sleep(1)
            st.rerun()

    # --- SINGLE DOCKING EXECUTION FOR MANUAL UPLOADS ---
    st.write("---")
    if st.button("🚀 Initialize Docking Algorithm (Single Run Mode)", type="secondary", disabled=not can_dock):
        vina_command = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"]
        progress_bar = st.progress(0, text="Initializing computational engine...")
        try:
            process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output_log, progress_count, current_line = [], 0, ""
            while True:
                char = process.stdout.read(1).decode("utf-8", errors="ignore")
                if not char: break
                output_log.append(char)
                if char == '*':
                    progress_count += 1
                    progress_bar.progress(min(100, int((progress_count / 50) * 100)), text="Exploring binding modes...")
            process.wait()
            if process.returncode == 0:
                progress_bar.progress(100, text="Optimization complete!")
                st.session_state.docking_results_raw = "".join(output_log)
                st.session_state.uff_cache = {} 
                time.sleep(0.5); st.rerun()
            else: st.error("Engine calculation error.")
        except Exception as e: st.error(f"Pipeline failed: {e}")

with col_visual:
    st.header("5. Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None and not st.session_state.get("comparative_run_complete", False):
        view_tabs = st.tabs(["Receptor & Grid Space", "Standalone Ligand (Interactive 3D)"])
        
        with view_tabs[0]:
            receptor_view_data = ""
            if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
                with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
            render_advanced_modeling_blueprint(receptor_view_data, st.session_state.serialized_ligand_block, mode="cartoon")
            
        with view_tabs[1]:
            if st.session_state.ligand_ready and st.session_state.serialized_ligand_block:
                st.markdown("### 🔬 Isolated Drug Topology")
                ligand_html = f"""
                <div id="ligand_container" style="height: 420px; width: 100%; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
                <script>
                    let viewer = $3Dmol.createViewer(document.getElementById('ligand_container'), {{backgroundColor: '#ffffff'}});
                    viewer.addModel(`{st.session_state.serialized_ligand_block}`, 'pdb'); 
                    viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.20}}, sphere: {{radius: 0.35}} }});
                    viewer.zoomTo(); viewer.render();
                </script>
                """
                components.html(ligand_html, height=450)
                
    elif st.session_state.docking_results_raw is not None and not st.session_state.get("comparative_run_complete", False):
        st.subheader("Interactive Single Run Complex Viewport")
        if os.path.exists("docking_poses.pdbqt"):
            parsed_poses = split_docking_poses("docking_poses.pdbqt")
            if parsed_poses:
                selected_pose = st.selectbox("Choose Docking Pose to Visualize:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Fit")
                if selected_pose not in parsed_poses: selected_pose = list(parsed_poses.keys())[0]

                with open("protein.pdbqt", "r") as f: protein_data = f.read()
                
                def get_pose_affinity(stdout_text, idx):
                    for line in stdout_text.split("\n"):
                        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
                        if m and int(m.group(1)) == idx: return m.group(2)
                    return "N/A"
                
                pose_affinity_score = get_pose_affinity(st.session_state.docking_results_raw, selected_pose)
                try:
                    aff_val = float(pose_affinity_score)
                    if aff_val > 0: aff_color = "#c62828"
                    elif aff_val <= -20.0: aff_color = "#d84315"; st.error(f"🚨 **Anomaly! Score: {aff_val}**")
                    else: aff_color = "#1b5e20"
                except ValueError: aff_color = "#1b5e20"

                cache_key = f"uff_{st.session_state.protein_name}_{selected_pose}"
                uff_progress_placeholder = st.empty() 
                
                if cache_key not in st.session_state.uff_cache:
                    try: pre_uff, post_uff, delta_uff = execute_uff_complex_minimization("protein.pdbqt", parsed_poses[selected_pose], uff_progress_placeholder)
                    except Exception as e: pre_uff, post_uff, delta_uff = "N/A", "N/A", "N/A"
                    st.session_state.uff_cache[cache_key] = (pre_uff, post_uff, delta_uff)
                
                uff_progress_placeholder.empty()
                pre_uff, post_uff, delta_uff = st.session_state.uff_cache[cache_key]

                active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
                
                res_str = ", ".join(list(set([i["Residue Contact"] for i in active_interactions]))) if active_interactions else "None"
                bond_str = ", ".join(list(set([i["Interaction Type"] for i in active_interactions]))) if active_interactions else "None"

                html_metric_card = f"""
                <div style="background-color:#f0f7f4; border-left:6px solid #2e7d32; padding:16px; border-radius:8px; margin-bottom:15px; font-family:sans-serif;">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e0e8e4; padding-bottom:8px; margin-bottom:10px;">
                        <div>
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold;">Active Pose Vina Affinity</span><br>
                            <span style="font-size:36px; font-weight:900; color:{aff_color};">{pose_affinity_score} <span style="font-size:18px; font-weight:normal;">kcal/mol</span></span>
                        </div>
                    </div>
                    <div style="font-size: 13px; color: #444; margin-bottom:8px;"><b>📍 UFF Initial Energy:</b> {pre_uff} kcal/mol | <b>📉 Optimized Energy:</b> {post_uff} kcal/mol</div>
                    <div style="font-size: 13px; color: #444; background-color:#e8f5e9; padding:10px; border-radius:5px;">
                        <b>🧬 Interacting Residues:</b> {res_str}<br>
                        <b>🔗 Contact Bond Types:</b> {bond_str}
                    </div>
                </div>
                """
                st.html(html_metric_card)

                col_render, col_mesh = st.columns([1, 1])
                with col_render: style_mode = re.sub(r'\W+', '', st.radio("Macromolecule Style Mode:", ["Cartoon Ribbon Mesh", "Sticks Profile"]).split()[0].lower())
                with col_mesh: surf_toggle = st.checkbox("Overlay Pocket Mesh", value=False)
                    
                render_advanced_modeling_blueprint(receptor_data=protein_data, ligand_data=parsed_poses[selected_pose], mode=style_mode, show_surface=surf_toggle, interactions_list=active_interactions)

                st.markdown("### 📋 Local Contact Residues & Bond Assignments Matrix")
                if active_interactions:
                    df_int = pd.DataFrame(active_interactions)
                    st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
                else:
                    st.info("No close contacts detected within a 3.8 Å threshold radius.")

# --- ADVANCED COMPARATIVE DOCKING RESULT PANEL ---
if st.session_state.get("comparative_run_complete", False):
    st.write("---")
    st.header("📊 5. Photopharmacology Analytical Report")
    
    t_aff = get_top_affinity_from_pdbqt("docking_trans.pdbqt")
    c_aff = get_top_affinity_from_pdbqt("docking_cis.pdbqt")
    n_aff = get_top_affinity_from_pdbqt("docking_native.pdbqt") if os.path.exists("docking_native.pdbqt") else t_aff
    
    delta_g_switch = round(c_aff - t_aff, 2)
    
    comp_data = {
        "Compound State": ["Native Scaffold (Original)", "Trans-Azo (Dark State / Active)", "Cis-Azo (Light State / Inactive)"],
        "Top Affinity (kcal/mol)": [n_aff, t_aff, c_aff],
        "ΔG vs Native": ["0.00", f"{round(t_aff - n_aff, 2):.2f}", f"{round(c_aff - n_aff, 2):.2f}"]
    }
    df_comp = pd.DataFrame(comp_data)
    
    col_tab, col_stmt = st.columns([1.5, 1])
    with col_tab:
        st.markdown("### Binding Affinity Comparison Matrix")
        st.dataframe(df_comp, hide_index=True, use_container_width=True)
        
    with col_stmt:
        if delta_g_switch > 1.5:
            verdict = f"**Successful Switch Effect:** The massive shape change caused a **+{delta_g_switch} kcal/mol** penalty in binding energy (loss of affinity vs trans). Light exposure successfully turns this drug 'OFF'."
            box_c, border_c = "#e8f5e9", "#2e7d32"
        elif delta_g_switch < -1.5:
            verdict = f"**Reverse Switch Effect:** The Cis isomer binds significantly better (**{delta_g_switch} kcal/mol** shift). Light exposure will turn this drug 'ON'."
            box_c, border_c = "#e3f2fd", "#1565c0"
        else:
            verdict = f"**Ineffective Switch:** Both isomers bind with similar affinity (ΔΔG: **{delta_g_switch} kcal/mol**). The pocket is too large or flexible to block the bent isomer."
            box_c, border_c = "#fff3e0", "#ef6c00"

        st.markdown(f"""
        <div style="background-color:{box_c}; border-left:6px solid {border_c}; padding:16px; border-radius:8px;">
            <h4 style="margin-top:0;">Automated Analysis</h4>
            <p style="font-size:14px; margin-bottom:5px;"><b>Trans (Active):</b> {t_aff} kcal/mol</p>
            <p style="font-size:14px; margin-bottom:10px;"><b>Cis (Inactive):</b> {c_aff} kcal/mol</p>
            <p style="font-size:14px; line-height:1.4;">{verdict}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🔬 Comparative 3D Binding Topologies")
    poses_t = split_docking_poses("docking_trans.pdbqt")
    poses_c = split_docking_poses("docking_cis.pdbqt")
    
    if os.path.exists("protein.pdbqt"):
        with open("protein.pdbqt", "r") as f: prot_data = f.read()
        
        int_t = compute_spatial_interactions("protein.pdbqt", poses_t[1]) if 1 in poses_t else []
        int_c = compute_spatial_interactions("protein.pdbqt", poses_c[1]) if 1 in poses_c else []
        
        if 1 in poses_t and 1 in poses_c:
            render_dual_docking_viewport(prot_data, poses_t[1], poses_c[1], int_t, int_c)
            
        col_t_table, col_c_table = st.columns(2)
        with col_t_table:
            st.markdown("#### Trans Isomer Interactions")
            if int_t:
                st.dataframe(pd.DataFrame(int_t)[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
            else: st.info("No close contacts detected.")
        with col_c_table:
            st.markdown("#### Cis Isomer Interactions")
            if int_c:
                st.dataframe(pd.DataFrame(int_c)[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
            else: st.info("No close contacts detected.")
            
    comp_html_report = f"""<!DOCTYPE html>
    <html>
    <head><title>PhotoDock Comparative Report</title></head>
    <body style="font-family: sans-serif; padding: 20px; background-color: #fcfcfc;">
        <h2 style="color: #2e7d32;">🔬 PhotoDock Comparative Analysis Report</h2>
        <p><b>Target Protein:</b> {st.session_state.protein_name}</p>
        <p><b>Base Azo-Scaffold:</b> {st.session_state.get('mutated_azo_smiles', 'N/A')}</p>
        <hr>
        <div style="background-color:{box_c}; border-left:6px solid {border_c}; padding:16px; border-radius:8px;">
            <h3>Comparative Statement (ΔΔG)</h3>
            <p><b>Trans Isomer Affinity (Dark State):</b> {t_aff} kcal/mol</p>
            <p><b>Cis Isomer Affinity (Light State):</b> {c_aff} kcal/mol</p>
            <p><b>Delta G Shift:</b> {delta_g_switch} kcal/mol</p>
            <p><b>Verdict:</b> {verdict.replace('**', '')}</p>
        </div>
    </body>
    </html>"""
    
    st.download_button(
        label="🔥 Download Comparative HTML Report",
        data=comp_html_report,
        file_name="PhotoDock_Comparative_Report.html",
        mime="text/html",
        use_container_width=True,
        type="primary"
    )
