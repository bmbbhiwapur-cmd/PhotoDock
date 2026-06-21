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

def get_top_affinity(stdout_text):
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*1\s+([-+]?\d+\.\d+)", line)
        if m: return float(m.group(1))
    return 0.0

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

def render_dual_docking_viewport(protein_data, trans_pose, cis_pose):
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
        v_t.zoomTo(); v_t.render();

        let v_c = $3Dmol.createViewer(document.getElementById('dock_cis'), {{backgroundColor: '#ffffff'}});
        v_c.addModel(`{protein_data}`, 'pdb');
        v_c.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
        v_c.addModel(`{cis_pose}`, 'pdb');
        v_c.setStyle({{model: 1}}, {{stick: {{colorscheme: 'orangeCarbon', radius: 0.28}}}});
        v_c.zoomTo(); v_c.render();
    </script>
    """
    components.html(html_content, height=470)

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
if "smiles_cache" not in st.session_state: st.session_state.smiles_cache = ""
if "mutated_azo_smiles" not in st.session_state: st.session_state.mutated_azo_smiles = ""
if "comparative_run_complete" not in st.session_state: st.session_state.comparative_run_complete = False

if st.button("🔄 Reset Entire Environment", type="secondary"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "docking_trans.pdbqt", "docking_cis.pdbqt", "temp_lig_state.pdb", "ligand_trans.pdbqt", "ligand_cis.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.rerun()

st.header("1. Target Protein Setup")
cancer_receptors = {
    "EGFR Kinase Domain (Lung/Breast Cancer)": "1M17",
    "Estrogen Receptor Alpha (Breast Cancer)": "3ERT",
    "BRAF V600E Mutant (Melanoma)": "4HJO",
    "GLUT4 Glucose Transporter (Diabetes)": "7WSM"
}
selected_receptor = st.selectbox("Select Target Receptor:", list(cancer_receptors.keys()))
pdb_id_input = cancer_receptors[selected_receptor]

if st.button("📥 Load Target Structure"):
    success, path = fetch_pdb_from_rcsb(pdb_id_input)
    if success:
        st.session_state.local_target_path = path
        meta = extract_pdb_metadata(path, pdb_id_input.upper())
        st.session_state.pdb_id_display = meta["id"]
        st.session_state.protein_name = meta["name"]
        conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt", allowed_heteroatoms=[])
        st.session_state.target_ready = conv_ok
        st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
    else: st.error(path)

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
    
    st.write("---")
    st.header("🚀 4. Execute Comparative Photo-Docking")
    st.markdown("This will run AutoDock Vina simultaneously on both the **Trans** (Dark State) and **Cis** (Light State) isomers in the exact same receptor cavity to compare binding drop-off.")
    
    if st.button("⚡ Run Comparative Docking (Trans vs Cis)", type="primary"):
        progress_bar = st.progress(0, text="Initializing Trans Isomer Docking...")
        
        # 1. Dock Trans
        cmd_trans = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_trans.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_trans.pdbqt"]
        t_proc = subprocess.run(cmd_trans, capture_output=True, text=True)
        st.session_state.raw_trans_log = t_proc.stdout
        
        progress_bar.progress(50, text="Trans Docking Complete. Initializing Cis Isomer Docking...")
        
        # 2. Dock Cis
        cmd_cis = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_cis.pdbqt", "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), "--exhaustiveness", str(exhaustiveness), "--out", "docking_cis.pdbqt"]
        c_proc = subprocess.run(cmd_cis, capture_output=True, text=True)
        st.session_state.raw_cis_log = c_proc.stdout
        
        progress_bar.progress(100, text="Comparative Docking Complete!")
        st.session_state.comparative_run_complete = True
        time.sleep(1)
        st.rerun()

if st.session_state.comparative_run_complete:
    st.write("---")
    st.header("📊 5. Photopharmacology Analytical Report")
    
    # Extract affinities
    t_aff = get_top_affinity(st.session_state.raw_trans_log)
    c_aff = get_top_affinity(st.session_state.raw_cis_log)
    
    delta_g = round(c_aff - t_aff, 2)
    
    # Dynamic Statement Logic
    if delta_g > 1.5:
        verdict = f"**Successful Switch Effect:** The massive shape change caused a **{delta_g} kcal/mol penalty** in binding energy. Light exposure successfully turns this drug 'OFF'."
        box_color = "#e8f5e9"
        border = "#2e7d32"
    elif delta_g < -1.5:
        verdict = f"**Reverse Switch Effect:** The Cis isomer binds better! Light exposure will turn this drug 'ON'. ΔΔG shift: **{delta_g} kcal/mol**."
        box_color = "#e3f2fd"
        border = "#1565c0"
    else:
        verdict = f"**Ineffective Switch:** Both isomers bind with similar affinity (ΔΔG: **{delta_g} kcal/mol**). The pocket is too large or flexible. A different switch position is required."
        box_color = "#fff3e0"
        border = "#ef6c00"

    st.markdown(f"""
    <div style="background-color:{box_color}; border-left:6px solid {border}; padding:16px; border-radius:8px; margin-bottom:20px;">
        <h3 style="margin-top:0;">Comparative Binding Statement (ΔΔG)</h3>
        <p style="font-size:16px;">The <b>Trans isomer (Dark State)</b> binds with an affinity of <b style="color:#2e7d32;">{t_aff} kcal/mol</b>, perfectly mimicking the native drug. 
        When triggered by light, it converts to the <b>Cis isomer</b>, altering the spatial geometry and shifting the affinity to <b style="color:#c62828;">{c_aff} kcal/mol</b>.</p>
        <p style="font-size:16px;">{verdict}</p>
    </div>
    """, unsafe_allow_html=True)

    # Render Dual 3D Viewport
    poses_t = split_docking_poses("docking_trans.pdbqt")
    poses_c = split_docking_poses("docking_cis.pdbqt")
    
    with open("protein.pdbqt", "r") as f: prot_data = f.read()
    
    if 1 in poses_t and 1 in poses_c:
        render_dual_docking_viewport(prot_data, poses_t[1], poses_c[1])
