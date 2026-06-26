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
import base64
from io import BytesIO

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
    for col in df.columns: html += f'<th style="background-color:#f2f2f2; color:#111;">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            val = row[col]
            if col == 'Affinity (kcal/mol)':
                try:
                    if float(val) > 0: html += f'<td style="color: #c62828; font-weight: bold;">{val}</td>'
                    else: html += f'<td style="color: #1b5e20; font-weight: bold;">{val}</td>'
                except: html += f'<td style="color:#111;">{val}</td>'
            else: html += f'<td style="color:#111;">{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

# --- VISUALIZATION CONSTRUCTS ---
def render_photopharmacology_tutorial():
    """Renders a custom HTML/JS interactive animation to explain the Cis/Trans shape shifting."""
    html_code = """
    <div style="font-family: sans-serif; background: #f0f7f4; padding: 20px; border-radius: 10px; border: 2px solid #2e7d32; text-align: center;">
        <h3 style="margin-top: 0; color: #111;">Interactive Mechanism: The Photopharmacology Switch</h3>
        <p style="font-size: 14px; color: #333;">Click the buttons below to see how the geometry affects receptor binding.</p>
        <div style="margin-bottom: 20px;">
            <button onclick="playAnim('native')" style="padding: 10px 16px; margin: 5px; cursor: pointer; background: #2e7d32; color: white; border: none; border-radius: 4px; font-weight:bold;">1. Native Scaffold (Original)</button>
            <button onclick="playAnim('trans')" style="padding: 10px 16px; margin: 5px; cursor: pointer; background: #1565c0; color: white; border: none; border-radius: 4px; font-weight:bold;">2. Trans-Azo (Dark State)</button>
            <button onclick="playAnim('cis')" style="padding: 10px 16px; margin: 5px; cursor: pointer; background: #c62828; color: white; border: none; border-radius: 4px; font-weight:bold;">3. Cis-Azo (Light State)</button>
        </div>
        <div style="position: relative; width: 100%; max-width: 500px; height: 300px; margin: 0 auto; background: white; border: 2px dashed #ccc; border-radius: 8px; overflow: hidden;">
            <div style="position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); width: 140px; height: 120px; border: 8px solid #555; border-top: none; border-radius: 0 0 30px 30px; background: #e0e0e0; box-shadow: inset 0px -10px 20px rgba(0,0,0,0.1);">
                <div style="text-align: center; margin-top: 50px; font-weight: bold; color: #111;">Receptor Cavity</div>
            </div>
            <div id="drug" style="position: absolute; top: 20px; left: 50%; transform: translateX(-50%); width: 30px; height: 100px; background: #2e7d32; transition: all 0.8s cubic-bezier(0.25, 0.8, 0.25, 1); transform-origin: center;"></div>
            <div id="status" style="position: absolute; top: 15px; width: 100%; text-align: center; font-weight: bold; font-size: 16px; color: #2e7d32; padding: 0 10px;"></div>
        </div>
        <script>
            function playAnim(state) {
                const drug = document.getElementById('drug');
                const status = document.getElementById('status');
                
                if (state === 'native') {
                    drug.style.background = '#2e7d32';
                    drug.style.height = '100px';
                    drug.style.width = '30px';
                    drug.style.marginLeft = '-15px';
                    drug.style.borderRadius = '15px';
                    drug.style.top = '120px'; 
                    status.innerText = "Native Scaffold: Fits Perfectly (Drug ON)";
                    status.style.color = "#2e7d32";
                } else if (state === 'trans') {
                    drug.style.background = '#1565c0';
                    drug.style.height = '100px';
                    drug.style.width = '30px';
                    drug.style.marginLeft = '-15px';
                    drug.style.borderRadius = '15px';
                    drug.style.top = '120px'; 
                    status.innerText = "Trans-Azo: Mimics Native Shape (Drug ON)";
                    status.style.color = "#1565c0";
                } else if (state === 'cis') {
                    drug.style.background = '#c62828';
                    drug.style.height = '50px'; 
                    drug.style.width = '120px'; 
                    drug.style.marginLeft = '-60px';
                    drug.style.borderRadius = '20px 20px 5px 5px';
                    drug.style.top = '70px'; 
                    status.innerText = "Cis-Azo: Bent Shape Clashes with Walls (Drug OFF)";
                    status.style.color = "#c62828";
                }
            }
            setTimeout(() => playAnim('native'), 500);
        </script>
    </div>
    """
    components.html(html_code, height=470)

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

def render_dual_docking_viewport(protein_data, trans_pose, cis_pose, int_t=[], int_c=[], mode="cartoon", show_surface=False):
    int_js_t = ""
    for interact in int_t:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_js_t += f"""
        v_t.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        v_t.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """
        
    int_js_c = ""
    for interact in int_c:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_js_c += f"""
        v_c.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        v_c.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """

    prot_style = "{cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}}"
    if mode == 'stick': prot_style = "{stick: {colorscheme: 'chain', radius:0.25}}"
    elif mode == 'spacefill': prot_style = "{sphere: {colorscheme: 'chain', radius:1.1}}"
    
    surf_js = "v_t.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0}); v_c.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""

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
        v_t.setStyle({{model: 0}}, {prot_style});
        v_t.addModel(`{trans_pose}`, 'pdb');
        v_t.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        {int_js_t}
        
        let v_c = $3Dmol.createViewer(document.getElementById('dock_cis'), {{backgroundColor: '#ffffff'}});
        v_c.addModel(`{protein_data}`, 'pdb');
        v_c.setStyle({{model: 0}}, {prot_style});
        v_c.addModel(`{cis_pose}`, 'pdb');
        v_c.setStyle({{model: 1}}, {{stick: {{colorscheme: 'orangeCarbon', radius: 0.28}}}});
        {int_js_c}
        
        {surf_js}
        
        v_t.zoomTo(); v_t.render();
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
if "smiles_cache" not in st.session_state: st.session_state.smiles_cache = ""
if "mutated_azo_smiles" not in st.session_state: st.session_state.mutated_azo_smiles = ""
if "comparative_run_complete" not in st.session_state: st.session_state.comparative_run_complete = False
if "uff_cache" not in st.session_state: st.session_state.uff_cache = {}
if "detected_pockets" not in st.session_state: st.session_state.detected_pockets = []
if "active_retained_ions" not in st.session_state: st.session_state.active_retained_ions = "None"
if "last_uploaded_protein" not in st.session_state: st.session_state.last_uploaded_protein = ""
if "last_uploaded_ligand" not in st.session_state: st.session_state.last_uploaded_ligand = "" 
if "selected_native_ligand" not in st.session_state: st.session_state.selected_native_ligand = "None"
if "ligand_summary_text" not in st.session_state: st.session_state.ligand_summary_text = ""
if "azologization_failed" not in st.session_state: st.session_state.azologization_failed = False

# Ensure Action Scopes
run_single_btn = False
run_comp_btn = False
can_dock = False

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

    # Updated Receptor Library with AMR Targets
    target_receptors = {
        "Cancer: EGFR Kinase Domain (Lung/Breast)": "1M17",
        "Cancer: Estrogen Receptor Alpha (Breast)": "3ERT",
        "Cancer: BRAF V600E Mutant (Melanoma)": "4HJO",
        "Diabetes: GLUT4 Glucose Transporter": "7WSM",
        "Antimicrobial: DNA Gyrase (S. aureus)": "4URO",
        "Antimicrobial: FtsZ Cell Division Protein": "3VOA",
        "Antimicrobial: DHFR (S. aureus)": "3SRW",
        "Antimicrobial: PBP2a (MRSA)": "1VQQ"
    }

    protein_source = st.radio("Choose Protein Input Method:", ["Select Curated Target Receptor", "Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    pdb_id_input = ""
    if protein_source == "Select Curated Target Receptor":
        selected_receptor = st.selectbox("Select Target Receptor:", list(target_receptors.keys()))
        pdb_id_input = target_receptors[selected_receptor]
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
            path = f"uploaded_{uploaded_file.name}"
            with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.session_state.local_target_path = path
            meta = extract_pdb_metadata(path, "Uploaded File")
            st.session_state.pdb_id_display = meta["id"]
            st.session_state.protein_name = meta["name"]
            
            if uploaded_file.name.lower().endswith(".pdb"):
                conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt", allowed_heteroatoms=[])
                st.session_state.target_ready = conv_ok
            else:
                os.replace(path, "protein.pdbqt")
                st.session_state.target_ready = True
                st.session_state.local_target_path = "protein.pdbqt"
            st.rerun()

    if st.session_state.target_ready and st.session_state.local_target_path:
        discovered_het = discover_and_list_all_heteroatoms(st.session_state.local_target_path)
        if discovered_het:
            st.subheader("🧬 Catalytic Cofactors Filter")
            selected_hets = []
            cols_het = st.columns(min(len(discovered_het), 4))
            for idx, (het_id, count) in enumerate(discovered_het.items()):
                with cols_het[idx % 4]:
                    if st.checkbox(f"Keep {het_id} ({count})", value=False, key=f"keep_het_{het_id}"):
                        selected_hets.append(het_id)
            if st.button("🛠 Rebuild Clean Receptor Structure"):
                ok, err = convert_pdb_to_pdbqt(st.session_state.local_target_path, "protein.pdbqt", is_ligand=False, allowed_heteroatoms=selected_hets)
                if ok: st.success("Receptor rebuilt!")
                else: st.error(f"Optimization failure: {err}")

    st.write("---")
    st.header("2. Phase 1: Small Molecule Phytochemical Setup")
    
    # Updated Phytochemical Library
    iks_library = {
        "Pterostilbene (Antidiabetic / Anticancer)": "COc1cc(C=Cc2ccc(O)cc2)cc(OC)c1",
        "Resveratrol (Antidiabetic / Anticancer)": "Oc1cc(O)cc(C=Cc2ccc(O)cc2)c1",
        "Combretastatin A-4 (Anticancer)": "COc1cc(C=Cc2ccc(O)c(OC)c2)cc(OC)c1OC",
        "Caffeic Acid (Antimicrobial)": "OC1=CC(=CC(=C1)O)C=CC(=O)O",
        "Ferulic Acid (Antimicrobial)": "COC1=C(C=CC(=C1)C=CC(=O)O)O",
        "Curcumin (Antimicrobial)": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
        "Psoralen (Antimicrobial - No C=C bridge)": "C1=CC2=C(C=C1)C3=C(C=CO3)OC2=O",
        "Marmelosin/Imperatorin (Antimicrobial - No bridge)": "CC(=CCOc1c2c(cc3c1oc(=O)cc3)cco2)C",
        "Plumbagin (Antimicrobial - No C=C bridge)": "CC1=CC(=O)C2=C(C1=O)C(=CC=C2)O",
        "Lawsone (Antimicrobial - No C=C bridge)": "O=C1C=C(O)C(=O)C2=CC=CC=C12",
        "Bergapten (Antimicrobial - No C=C bridge)": "COC1=C2C(=CC3=C1OC(=O)C=C3)C=CO2"
    }
    selected_phytochemical = st.selectbox("Select Native Scaffold:", list(iks_library.keys()))
    smiles_input_val = iks_library[selected_phytochemical]

    if st.button("📥 Load Ligand Structure"):
        pub_data = fetch_ligand_data_from_pubchem(smiles_input_val)
        ok, _ = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
        if ok:
            shutil.copy("ligand.pdbqt", "ligand_native.pdbqt")
            st.session_state.ligand_ready = True
            st.session_state.smiles_cache = smiles_input_val
            with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
            st.success(f"Loaded: {pub_data['name']}")
            st.rerun()

    if st.session_state.ligand_ready and st.session_state.smiles_cache:
        if st.checkbox("👁️ Show 2D Structural Native Representation", value=False):
            try:
                native_mol = Chem.MolFromSmiles(st.session_state.smiles_cache)
                AllChem.Compute2DCoords(native_mol)
                img = Draw.MolToImage(native_mol, size=(400, 300), fitImage=True)
                st.image(img, caption="Native 2D Scaffold")
            except Exception: st.error("Could not render 2D structure.")

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
                        st.session_state.azologization_failed = False
                        if os.path.exists("ligand_trans.pdbqt"):
                            shutil.copy("ligand_trans.pdbqt", "ligand.pdbqt")
                            with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                    else: 
                        st.error("Failed to map constrained 3D coordinates.")
                        st.session_state.azologization_failed = True
                else: 
                    st.warning("No valid free C=C bridge found in this molecule. Note: Compounds like Plumbagin and Lawsone require an external synthetic azo-linker, as they lack a native stilbene bridge.")
                    with st.expander("❓ Why did this happen? (Understanding Azologization & Rigid Geometry)", expanded=False):
                        st.markdown("""
                        **1. The Requirement of a Molecular "Hinge"**
                        For a molecule to dramatically change its shape when hit by a photon (isomerizing from an elongated *trans* state to a bent *cis* state), it requires a flexible, acyclic (non-ring) "hinge."
                        
                        The most common hinges used in drug design are:
                        * **Azobenzenes** (–N=N– bridge)
                        * **Stilbenes** (–C=C– bridge)
                        
                        When light hits these linear bridges, the double bond is temporarily excited and twists, physically bending the molecule by roughly 40 to 50 degrees. For this to work, the double bond must be free to rotate in 3D space.

                        **2. The Rigid Geometry of Plumbagin & Lawsone**
                        Plumbagin and Lawsone belong to a class of compounds called naphthoquinones. Their structures consist of two fused, highly planar six-membered rings (a benzene ring fused to a quinone ring).
                        
                        While these molecules do contain double bonds (both C=C and C=O), all of these double bonds are **locked inside a rigid, closed ring system.** Geometrically, a double bond inside a standard six-membered ring is permanently locked in the *cis* conformation relative to the ring's continuous chain.
                        
                        If you attempt to force a double bond inside a six-membered ring to flip to a *trans* configuration, the ring would physically shatter due to immense geometric strain.
                        
                        **"Azologization"** is a specific synthetic strategy where a chemist (or an algorithm) looks for an existing, free-spinning C=C double bond in a drug and replaces it with an N=N azobenzene bond. Because Plumbagin and Lawsone lack a free, linear alkene chain sticking out of the molecule, the algorithm correctly throws an error: there is no native bridge to replace.

                        **3. The Solution: External Synthetic Linkers (Tethering)**
                        When a drug candidate has a rigid core but possesses excellent medicinal properties, photopharmacologists use a different strategy called **Pendant Photoswitching (or tethering)**.
                        
                        Instead of trying to bend the rigid core of the molecule itself, chemists synthetically attach an external azobenzene "tail" to one of the molecule's existing functional groups (such as substituting the hydrogen on the –OH group of Lawsone).
                        """)
                    st.session_state.has_isomers = False
                    st.session_state.azologization_failed = True

    # 2D Isomers View right below Phase 3
    if st.session_state.get("has_isomers", False):
        st.markdown("### 2D Cis/Trans Geometric View")
        if st.checkbox("👁️ Show 2D Comparison", value=True):
            engine = Azologizer()
            col_2d_trans, col_2d_cis = st.columns(2)
            try:
                with col_2d_trans: st.image(engine.get_2d_isomer_image(st.session_state.mutated_azo_smiles, "trans"), caption="2D Trans Isomer")
                with col_2d_cis: st.image(engine.get_2d_isomer_image(st.session_state.mutated_azo_smiles, "cis"), caption="2D Cis Isomer")
            except Exception: pass

    # --- GEOMETRIC CAVITY SEARCH SITE PANEL ---
    if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
        st.write("---")
        st.header("3. Smart Cavity Pocket Finder")
        if st.button("🔍 Scan Surface For Structural Cavities", use_container_width=True):
            with st.spinner("Analyzing macromolecular spatial curvature dynamics..."):
                pockets_discovered = identify_protein_cavities("protein.pdbqt")
                st.session_state.detected_pockets = pockets_discovered
                if pockets_discovered: st.success(f"Successfully mapped {len(pockets_discovered)} surface cavities!")

        if st.session_state.detected_pockets:
            p_opts = st.session_state.detected_pockets
            selected_p_idx = st.selectbox("Select Target Computational Cavity:", options=range(len(p_opts)), format_func=lambda idx: f"{p_opts[idx]['Pocket_ID']} (Density Score: {p_opts[idx]['Score']})")
            if st.button("🎯 Align Grid Parameters to This Cavity"):
                chosen_p = p_opts[selected_p_idx]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_p["cx"], chosen_p["cy"], chosen_p["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_p["bx"], chosen_p["by"], chosen_p["bz"]
                st.success(f"Grid coordinates targeted over pocket space!")
                st.rerun()

    st.write("---")
    st.header("4. Grid Box Mechanics")
    
    # Circuit breaker: Docking is ONLY allowed if target/ligand are ready AND azologization didn't fail
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready and not st.session_state.azologization_failed)

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

    st.write("---")
    st.header("🚀 5. Execute Docking")
    
    # Render error state if azologization failed blocking the engine
    if st.session_state.azologization_failed:
        st.error("🚫 Docking Engine Locked: Azologization failed. Please select a chemically compatible phytochemical with a valid C=C bridge to proceed with photopharmacology docking.")
    
    run_single_btn = st.button("Initialize Docking Algorithm (Single Run Mode)", type="secondary", disabled=not can_dock)
    
    st.markdown("Run AutoDock Vina sequentially on **Native**, **Trans**, and **Cis** to construct a comparative matrix.")
    run_comp_btn = st.button("⚡ Run Full Comparative Docking Sequence", type="primary", disabled=not can_dock)

# --- DOCKING EXECUTION LOGIC ---
if run_single_btn and can_dock:
    st.session_state.comparative_run_complete = False
    vina_command = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"]
    progress_bar = st.progress(0, text="Initializing computational engine...")
    try:
        process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_log, progress_count = [], 0
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

elif run_comp_btn and can_dock:
    # 1. Setup Placeholders for Animation and Progress
    anim_placeholder = st.empty()
    progress_bar = st.progress(0, text="Initializing comparative sequence...")

    def trigger_light_state(state):
        if state == "native":
            bg, color, msg = "#e8f5e9", "#2e7d32", "🌿 STAGE 1: Docking Native Scaffold (Baseline)"
        elif state == "dark":
            bg, color, msg = "#111111", "#4caf50", "🌑 STAGE 2 (DARK STATE): Docking Trans-Isomer..."
        elif state == "light":
            bg, color, msg = "#ffebee", "#c62828", "💡 STAGE 3 (PHOTON PULSE): Docking Cis-Isomer..."
            
        html = f"""
        <div style="background:{bg}; padding:25px; border-radius:10px; border:3px solid {color}; text-align:center; transition: background-color 1s ease;">
            <h2 style="color:{color}; margin:0; animation: pulse 1.5s infinite;">{msg}</h2>
            <style>@keyframes pulse {{ 0% {{opacity: 0.5;}} 50% {{opacity: 1;}} 100% {{opacity: 0.5;}} }}</style>
        </div>
        """
        anim_placeholder.markdown(html, unsafe_allow_html=True)

    # 2. Native Docking
    trigger_light_state("native")
    progress_bar.progress(10, text="Docking Native Ligand...")
    if os.path.exists("ligand_native.pdbqt"):
        cmd_native = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_native.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_native.pdbqt"]
        subprocess.run(cmd_native, capture_output=True, text=True)
        
    # 3. Trans Docking (Dark State)
    trigger_light_state("dark")
    progress_bar.progress(40, text="Docking Trans Isomer (Dark State)...")
    if os.path.exists("ligand_trans.pdbqt"):
        cmd_trans = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_trans.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_trans.pdbqt"]
        t_proc = subprocess.run(cmd_trans, capture_output=True, text=True)
        st.session_state.raw_trans_log = t_proc.stdout
    
    # 4. Cis Docking (Light State)
    trigger_light_state("light")
    progress_bar.progress(75, text="Docking Cis Isomer (Light State)...")
    if os.path.exists("ligand_cis.pdbqt"):
        cmd_cis = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_cis.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_cis.pdbqt"]
        c_proc = subprocess.run(cmd_cis, capture_output=True, text=True)
        st.session_state.raw_cis_log = c_proc.stdout
    
    # 5. Cleanup and State Update
    progress_bar.progress(100, text="Comparative Docking Complete!")
    anim_placeholder.empty() # Remove the animation box
    
    st.session_state.comparative_run_complete = True
    time.sleep(1)
    st.rerun()

# --- RIGHT COLUMN UI RENDERING ---
with col_visual:
    st.header("6. Active Viewport Canvas")
    
    if not st.session_state.target_ready and not st.session_state.ligand_ready:
        render_photopharmacology_tutorial()
    
    if st.session_state.docking_results_raw is None and not st.session_state.get("comparative_run_complete", False):
        if st.session_state.get("has_isomers", False):
            st.markdown("### 🔬 3D Structural Geometric Comparison")
            with open("ligand_trans.pdbqt", "r") as f: t_data = f.read()
            with open("ligand_cis.pdbqt", "r") as f: c_data = f.read()
            render_isomer_comparison(t_data, c_data)
        else:
            view_tabs = st.tabs(["Receptor Space", "Standalone Ligand"])
            with view_tabs[0]:
                receptor_view_data = ""
                if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
                    with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
                render_advanced_modeling_blueprint(receptor_view_data, st.session_state.serialized_lig
