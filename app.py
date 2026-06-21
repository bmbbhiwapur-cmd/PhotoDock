# app.py - PhotoDock IKS Professional Platform (Vina + UFF Integrated)
# Author: Sarang Dhote, Assistant Professor, Department of Chemistry
# Professional molecular docking platform integrating IKS with photopharmacology

import streamlit as st
import os
import sys
import subprocess
import time
import urllib.request
import pandas as pd
import numpy as np
import re
from pathlib import Path
from io import BytesIO
import base64

import matplotlib
matplotlib.use('Agg')

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Crippen

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="PhotoDock IKS | Real Molecular Docking",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CLOUD ENGINE BOOTSTRAP ====================
@st.cache_resource
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        try:
            url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
            urllib.request.urlretrieve(url, binary_name)
            os.chmod(binary_name, 0o755)
            return True
        except Exception as e:
            return False
    return True

vina_ready = ensure_linux_vina_exists()

# ==================== IKS LIGAND DATABASE (WITH REAL ISOMERS) ====================
IKS_LIGANDS = {
    "Plumbagin (Chitraka - Plumbago zeylanica)": {
        "smiles_dark": "CC1=CC(=O)c2c(C1=O)cccc2O", 
        "smiles_light": "CC1=CC(O)c2c(C1O)cccc2O", # Photo-reduced hydroquinone state
        "iupac": "5-hydroxy-2-methylnaphthalene-1,4-dione",
        "iks_use": "व्रणशोधन (Wound cleansing) — Root paste applied with sun exposure",
        "target": "DNA Gyrase",
        "mechanism": "DNA intercalation inhibitor",
    },
    "Marmelosin (Bilva - Aegle marmelos)": {
        "smiles_dark": "CC(=CCOc1c2C=CC(=O)Oc2cc3c1C=CO3)C", # Trans
        "smiles_light": "C/C(C)=C\\COc1c2C=CC(=O)Oc2cc3c1C=CO3", # Cis photoisomer
        "iupac": "9-(3-methylbut-2-enoxy)furo[3,2-g]chromen-7-one",
        "iks_use": "श्वित्र (Vitiligo) — Leaf juice applied with morning sunlight",
        "target": "FtsZ Protein",
        "mechanism": "FtsZ polymerization inhibitor",
    },
    "Lawsone (Mendhi - Lawsonia inermis)": {
        "smiles_dark": "O=C1C=CC(=O)c2ccccc12",
        "smiles_light": "OC1=CC=C(O)c2ccccc12", # Photo-reduced state
        "iupac": "2-hydroxynaphthalene-1,4-dione",
        "iks_use": "कुष्ठघ्न (Antifungal) — Henna paste sun-dried on skin",
        "target": "Cytochrome bd Oxidase",
        "mechanism": "Electron transport chain disruptor",
    },
    "Psoralen (Bakuchi - Psoralea corylifolia)": {
        "smiles_dark": "O=C1C=CC2=CC=CC3=C2C1=CO3",
        "smiles_light": "O=C(O)/C=C/C1=CC=CC2=C1C=CO2", # UV Ring-opened intermediate
        "iupac": "furo[3,2-g]chromen-7-one",
        "iks_use": "श्वित्रहर (Vitiligo cure) — Seed powder with morning sun",
        "target": "DNA (Thymidine cross-linking)",
        "mechanism": "UV-activated DNA photoadduct formation",
    }
}

# ==================== RECEPTOR DATABASE ====================
RECEPTORS = {
    "DNA Gyrase B (S. aureus) - PDB: 2XCS": {
        "pdb_id": "2XCS",
        "organism": "Staphylococcus aureus",
        "function": "ATP-dependent DNA supercoiling essential for replication",
        "grid_center": [15.5, 32.1, 22.7],
        "grid_size": [22, 22, 22],
    },
    "FtsZ (S. aureus) - PDB: 4DXD": {
        "pdb_id": "4DXD",
        "organism": "Staphylococcus aureus",
        "function": "Tubulin-like GTPase for bacterial cell division",
        "grid_center": [28.3, 15.9, 42.5],
        "grid_size": [22, 22, 22],
    },
    "PBP2a (MRSA) - PDB: 6V5S": {
        "pdb_id": "6V5S",
        "organism": "Methicillin-resistant Staphylococcus aureus",
        "function": "Cell wall transpeptidase conferring β-lactam resistance",
        "grid_center": [42.1, 11.8, 35.2],
        "grid_size": [24, 24, 24],
    },
}

# ==================== PROFESSIONAL CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tiro+Devanagari+Sanskrit&family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .sanskrit-text {
        font-family: 'Tiro Devanagari Sanskrit', serif;
        font-size: 1.4em;
        color: #8B0000;
        text-align: center;
        padding: 12px;
        background: linear-gradient(135deg, #fff8e1, #fff3e0);
        border-radius: 10px;
        border: 1px solid #ffcc80;
        margin: 10px 0;
    }
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    .metric-value { font-size: 2.2em; font-weight: 700; color: #1a237e; }
    .metric-label { color: #666; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session():
    defaults = {
        'current_step': 1,
        'selected_ligand': None,
        'selected_receptor': None,
        'dark_affinity': None,
        'dark_poses_file': None,
        'dark_uff': None,
        'light_affinity': None,
        'light_poses_file': None,
        'light_uff': None,
        'current_isomer': 'dark',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ==================== BIOINFORMATICS PIPELINE ====================
def fetch_pdb_from_rcsb(pdb_id):
    local_pdb = f"{pdb_id}.pdb"
    if not os.path.exists(local_pdb):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, local_pdb)
    return local_pdb

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    try:
        with open(input_pdb, "r", encoding="utf-8") as pdb, open(output_pdbqt, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    res_name = line[17:20].strip()
                    if record_type := line[:6].strip() == "HETATM" and not is_ligand and res_name in ["HOH", "WAT"]: continue
                    
                    atom_name, chain_id = line[12:16], line[21].strip() or "A"
                    try: x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    except ValueError: continue
                    
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    vina_type = element.upper() if element.upper() in ["A", "C", "N", "O", "S", "P"] else "A"
                    
                    pdbqt.write(f"{line[:6]:<6}{line[6:11]:>5} {atom_name:<4} {res_name:>3} {chain_id}{line[22:26]:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00    +0.000 {vina_type:<2}\n")
            if is_ligand: 
                pdbqt.write("ENDROOT\nTORSDOF 4\n")
            else: pdbqt.write("ENDMDL\n")
        return True
    except Exception as e: return False

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        return True
    except Exception as e: return False

def execute_uff_complex_minimization(protein_path, ligand_pose_str):
    try:
        protein_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
        ligand_mol = Chem.MolFromPDBBlock(ligand_pose_str, sanitize=False, removeHs=False)
        combined = Chem.CombineMols(protein_mol, ligand_mol)
        Chem.SanitizeMol(combined, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        
        uff_field = AllChem.UFFGetMoleculeForceField(combined)
        pre_energy = uff_field.CalcEnergy()
        uff_field.Minimize(maxIts=200, forceTol=1e-3)
        post_energy = uff_field.CalcEnergy()
        
        return round(pre_energy, 2), round(post_energy, 2), round(post_energy - pre_energy, 2)
    except Exception:
        return "N/A", "N/A", "N/A"

def parse_vina_log(log_text):
    for line in log_text.split('\n'):
        m = re.match(r"^\s*1\s+([-+]?\d+\.\d+)", line)
        if m: return float(m.group(1))
    return 0.0

def extract_first_pose(pdbqt_file):
    pose_lines = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            pose_lines.append(line)
            if line.startswith("ENDMDL"): break
    return "".join(pose_lines)

def render_3d_viewer(receptor_path, ligand_block, height=400):
    with open(receptor_path, 'r') as f: rec_data = f.read()
    rec_esc = rec_data.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    lig_esc = ligand_block.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    
    html = f"""
    <div id="viewer" style="width:100%; height:{height}px; position:relative; background:#f5f5f5; border-radius:8px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{backgroundColor:'white'}});
        viewer.addModel(`{rec_esc}`, "pdb");
        viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval'}}}});
        viewer.addModel(`{lig_esc}`, "pdb");
        viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.3}}}});
        viewer.zoomTo(); viewer.render();
    </script>
    """
    st.components.v1.html(html, height=height+20)

# ==================== UI WORKFLOW ====================
st.markdown("""
<div class="main-header">
    <h1>💡 PhotoDock IKS</h1>
    <h3>Real Computational Photopharmacology Platform</h3>
</div>
""", unsafe_allow_html=True)

if not vina_ready:
    st.error("🚨 Linux Vina Engine failed to mount. App cannot run docking.")
    st.stop()

# ----------------- STEP 1 -----------------
if st.session_state.current_step == 1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌿 Select IKS Photosensitizer")
        lig_key = st.selectbox("Phytochemical", list(IKS_LIGANDS.keys()))
        lig_data = IKS_LIGANDS[lig_key]
        st.info(f"**Traditional Use:** {lig_data['iks_use']}")
        
    with col2:
        st.markdown("#### 🦠 Select Target Receptor")
        rec_key = st.selectbox("Bacterial Protein", list(RECEPTORS.keys()))
        rec_data = RECEPTORS[rec_key]
        st.info(f"**Target Function:** {rec_data['function']}")

    if st.button("🚀 Prepare Structures & Execute Dark State Docking", type="primary", use_container_width=True):
        st.session_state.selected_ligand = lig_key
        st.session_state.selected_receptor = rec_key
        
        with st.spinner("Downloading RCSB PDB & Compiling Dark State Topology..."):
            pdb_path = fetch_pdb_from_rcsb(rec_data['pdb_id'])
            convert_pdb_to_pdbqt(pdb_path, "protein.pdbqt")
            convert_smiles_to_pdbqt(lig_data['smiles_dark'], "ligand_dark.pdbqt")
            
        with st.spinner("Executing AutoDock Vina (Ground State)..."):
            cmd = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_dark.pdbqt",
                   "--center_x", str(rec_data['grid_center'][0]), "--center_y", str(rec_data['grid_center'][1]), "--center_z", str(rec_data['grid_center'][2]),
                   "--size_x", str(rec_data['grid_size'][0]), "--size_y", str(rec_data['grid_size'][1]), "--size_z", str(rec_data['grid_size'][2]),
                   "--exhaustiveness", "8", "--out", "dock_dark.pdbqt"]
            process = subprocess.run(cmd, capture_output=True, text=True)
            st.session_state.dark_affinity = parse_vina_log(process.stdout)
            st.session_state.dark_poses_file = "dock_dark.pdbqt"
            
        with st.spinner("Running UFF Complex Relaxation..."):
            pose1 = extract_first_pose("dock_dark.pdbqt")
            pre, post, delta = execute_uff_complex_minimization("protein.pdbqt", pose1)
            st.session_state.dark_uff = {"pre": pre, "post": post, "delta": delta}
            
        st.session_state.current_step = 2
        st.rerun()

# ----------------- STEP 2 -----------------
elif st.session_state.current_step == 2:
    st.markdown("### 🌑 Dark State Docking Complete")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Ground State Affinity</div>
            <div class="metric-value">{st.session_state.dark_affinity} kcal/mol</div>
            <hr>
            <div class="metric-label">UFF Steric Relaxation</div>
            <div style="text-align:center; color:#c62828; font-weight:bold;">{st.session_state.dark_uff['delta']} kcal/mol</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sanskrit-text" style="font-size:1.1em;">सूर्यसंयोग: (Sūrya-saṃyoga)</div>
        <p style="font-size:0.9em; color:#555;">Ready to bombard the target with UV photons to force isomerization into the active, higher-energy topological state.</p>
        """, unsafe_allow_html=True)
        
        if st.button("💥 Fire UV Photons (Isomerize & Re-Dock)", type="primary", use_container_width=True):
            lig_data = IKS_LIGANDS[st.session_state.selected_ligand]
            rec_data = RECEPTORS[st.session_state.selected_receptor]
            
            with st.spinner("Isomerizing topology & generating photo-adduct..."):
                convert_smiles_to_pdbqt(lig_data['smiles_light'], "ligand_light.pdbqt")
                
            with st.spinner("Executing AutoDock Vina (Photo-Activated State)..."):
                cmd = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_light.pdbqt",
                       "--center_x", str(rec_data['grid_center'][0]), "--center_y", str(rec_data['grid_center'][1]), "--center_z", str(rec_data['grid_center'][2]),
                       "--size_x", str(rec_data['grid_size'][0]), "--size_y", str(rec_data['grid_size'][1]), "--size_z", str(rec_data['grid_size'][2]),
                       "--exhaustiveness", "8", "--out", "dock_light.pdbqt"]
                process = subprocess.run(cmd, capture_output=True, text=True)
                st.session_state.light_affinity = parse_vina_log(process.stdout)
                st.session_state.light_poses_file = "dock_light.pdbqt"
                
            with st.spinner("Running UFF Complex Relaxation..."):
                pose1 = extract_first_pose("dock_light.pdbqt")
                pre, post, delta = execute_uff_complex_minimization("protein.pdbqt", pose1)
                st.session_state.light_uff = {"pre": pre, "post": post, "delta": delta}
                
            st.session_state.current_step = 3
            st.rerun()
            
    with col2:
        pose_str = extract_first_pose(st.session_state.dark_poses_file)
        render_3d_viewer("protein.pdbqt", pose_str)

# ----------------- STEP 3 -----------------
elif st.session_state.current_step == 3:
    st.markdown("### ☀️ Photoisomerization Analysis Complete")
    
    dark_aff = st.session_state.dark_affinity
    light_aff = st.session_state.light_affinity
    delta_aff = round(dark_aff - light_aff, 2)
    fold = 10 ** (delta_aff / 1.36) if delta_aff > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>🌑 Dark State</h4>
            <div class="metric-value">{dark_aff}</div>
            <div class="metric-label">kcal/mol</div>
            <p style="text-align:center; font-size:0.8em; color:#666;">UFF Delta: {st.session_state.dark_uff['delta']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid #fbc02d;">
            <h4>☀️ Light State (Activated)</h4>
            <div class="metric-value">{light_aff}</div>
            <div class="metric-label">kcal/mol</div>
            <p style="text-align:center; font-size:0.8em; color:#666;">UFF Delta: {st.session_state.light_uff['delta']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card" style="background:#e8f5e9;">
            <h4>📊 Selectivity</h4>
            <div class="metric-value">{delta_aff}</div>
            <div class="metric-label">ΔΔG (kcal/mol)</div>
            <p style="text-align:center; font-weight:bold; color:#2e7d32;">~{fold:.1f}x Fold Change</p>
        </div>
        """, unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Dark State Topology**")
        render_3d_viewer("protein.pdbqt", extract_first_pose(st.session_state.dark_poses_file), height=350)
    with v2:
        st.markdown("**Activated Light State Topology**")
        render_3d_viewer("protein.pdbqt", extract_first_pose(st.session_state.light_poses_file), height=350)

    st.write("---")
    if st.button("🔄 Start New Experiment", use_container_width=True):
        init_session()
        st.session_state.current_step = 1
        st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-family: 'Tiro Devanagari Sanskrit', serif; font-size:1.2em; color:#8B0000;">सूर्यसंयोगचिकित्सा</p>
    <p>💡 <b>PhotoDock IKS</b> — Real Vina Execution & RDKit UFF Relaxation</p>
    <p style="font-size:0.85em; color:#666;"><b>Sarang Dhote</b>, Assistant Professor, Department of Chemistry</p>
</div>
""", unsafe_allow_html=True)
