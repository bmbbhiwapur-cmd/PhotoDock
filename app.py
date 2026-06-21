# app.py - PhotoDock IKS | Integrated Vina & Dravyaguna Platform
# Author: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry

import streamlit as st
import os
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
from rdkit.Chem import AllChem, Draw, Descriptors

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
        except Exception:
            return False
    return True

vina_ready = ensure_linux_vina_exists()

# ==================== IKS LIGAND DATABASE (WITH DRAVYAGUNA & REAL ISOMERS) ====================
IKS_LIGANDS = {
    "Plumbagin (Chitraka - Plumbago zeylanica)": {
        "smiles_dark": "CC1=CC(=O)c2c(C1=O)cccc2O", 
        "smiles_light": "CC1=CC(O)c2c(C1O)cccc2O", 
        "iupac": "5-hydroxy-2-methylnaphthalene-1,4-dione",
        "shloka": "चित्रको वह्निसदृशः पाचकः दीपनो लघुः।",
        "dravyaguna": "Rasa: Katu | Virya: Ushna | Vipaka: Katu",
        "karma": "Vrana-shodhana (Wound cleansing), Kushtaghna",
        "target": "DNA Gyrase B (S. aureus)",
    },
    "Marmelosin (Bilva - Aegle marmelos)": {
        "smiles_dark": "CC(=CCOc1c2C=CC(=O)Oc2cc3c1C=CO3)C", 
        "smiles_light": "C/C(C)=C\\COc1c2C=CC(=O)Oc2cc3c1C=CO3", 
        "iupac": "9-(3-methylbut-2-enoxy)furo[3,2-g]chromen-7-one",
        "shloka": "बिल्वं कषायं मधुरं पाचकं दीपनं लघु।",
        "dravyaguna": "Rasa: Kashaya, Tikta | Virya: Ushna | Vipaka: Katu",
        "karma": "Krimighna (Antimicrobial), Shvitra (Vitiligo repigmentation)",
        "target": "FtsZ Protein (S. aureus)",
    },
    "Lawsone (Mendhi - Lawsonia inermis)": {
        "smiles_dark": "O=C1C=CC(=O)c2ccccc12",
        "smiles_light": "OC1=CC=C(O)c2ccccc12", 
        "iupac": "2-hydroxynaphthalene-1,4-dione",
        "shloka": "मदयन्तिका कषाया च तिक्ता शीता च कुष्ठनुत्।",
        "dravyaguna": "Rasa: Tikta, Kashaya | Virya: Sheeta | Vipaka: Katu",
        "karma": "Kushtaghna (Antifungal), Dahaprashamana",
        "target": "PBP2a (MRSA)",
    },
    "Psoralen (Bakuchi - Psoralea corylifolia)": {
        "smiles_dark": "O=C1C=CC2=CC=CC3=C2C1=CO3",
        "smiles_light": "O=C(O)/C=C/C1=CC=CC2=C1C=CO2", 
        "iupac": "furo[3,2-g]chromen-7-one",
        "shloka": "बाकुची मधुरा तिक्ता कटुपाका रसायनी।",
        "dravyaguna": "Rasa: Tikta, Katu | Virya: Ushna | Vipaka: Katu",
        "karma": "Shvitrahara (Vitiligo cure), Kushtaghna",
        "target": "DNA Gyrase B (S. aureus)",
    }
}

# ==================== RECEPTOR DATABASE ====================
RECEPTORS = {
    "DNA Gyrase B (S. aureus)": {
        "pdb_id": "2XCS", "grid_center": [15.5, 32.1, 22.7], "grid_size": [22, 22, 22],
        "function": "ATP-dependent DNA supercoiling essential for bacterial replication"
    },
    "FtsZ Protein (S. aureus)": {
        "pdb_id": "4DXD", "grid_center": [28.3, 15.9, 42.5], "grid_size": [22, 22, 22],
        "function": "Tubulin-like GTPase required for bacterial cell division"
    },
    "PBP2a (MRSA)": {
        "pdb_id": "6V5S", "grid_center": [42.1, 11.8, 35.2], "grid_size": [24, 24, 24],
        "function": "Cell wall transpeptidase conferring profound β-lactam resistance"
    },
}

# ==================== STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tiro+Devanagari+Sanskrit&family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .card { background: #ffffff; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; margin-bottom: 15px;}
    .ayurveda-card { background: #f8fafc; border-left: 5px solid #2e7d32; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .shloka { font-family: 'Tiro Devanagari Sanskrit', serif; font-size: 1.4em; color: #1b5e20; text-align: center; margin: 10px 0;}
    .metric-value { font-size: 2.2em; font-weight: 700; color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'current_step' not in st.session_state:
    st.session_state.update({
        'current_step': 1, 'selected_ligand': None, 'selected_receptor': None,
        'dark_affinity': None, 'dark_poses': None, 'dark_uff': None,
        'light_affinity': None, 'light_poses': None, 'light_uff': None
    })

# ==================== HELPER FUNCTIONS ====================
def fetch_pdb_from_rcsb(pdb_id):
    local_pdb = f"{pdb_id}.pdb"
    if not os.path.exists(local_pdb):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", local_pdb)
    return local_pdb

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    try:
        with open(input_pdb, "r", encoding="utf-8") as pdb, open(output_pdbqt, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    res_name = line[17:20].strip()
                    if line[:6].strip() == "HETATM" and not is_ligand and res_name in ["HOH", "WAT"]: continue
                    atom_name, chain_id = line[12:16], line[21].strip() or "A"
                    try: x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    except ValueError: continue
                    element = line[76:78].strip() or ''.join([c for c in atom_name if c.isalpha()])[0]
                    vina_type = element.upper() if element.upper() in ["A", "C", "N", "O", "S", "P"] else "A"
                    pdbqt.write(f"{line[:6]:<6}{line[6:11]:>5} {atom_name:<4} {res_name:>3} {chain_id}{line[22:26]:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00    +0.000 {vina_type:<2}\n")
            if is_ligand: pdbqt.write("ENDROOT\nTORSDOF 4\n")
            else: pdbqt.write("ENDMDL\n")
        return True
    except Exception: return False

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    temp_pdb = "temp_ligand.pdb"
    Chem.MolToPDBFile(mol, temp_pdb)
    convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)

def execute_uff_complex_minimization(protein_path, ligand_pose_str):
    try:
        p_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
        l_mol = Chem.MolFromPDBBlock(ligand_pose_str, sanitize=False, removeHs=False)
        combined = Chem.CombineMols(p_mol, l_mol)
        Chem.SanitizeMol(combined, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        uff = AllChem.UFFGetMoleculeForceField(combined)
        pre = uff.CalcEnergy()
        uff.Minimize(maxIts=200, forceTol=1e-3)
        return round(pre, 2), round(uff.CalcEnergy(), 2)
    except Exception: return "N/A", "N/A"

def get_2d_image_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(350, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return ""

def render_3d_viewer(receptor_path, ligand_block, height=400, color="greenCarbon"):
    with open(receptor_path, 'r') as f: rec_data = f.read()
    r_esc = rec_data.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    l_esc = ligand_block.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    
    html = f"""
    <div id="v_{height}" style="width:100%; height:{height}px; background:#f8fafc; border-radius:8px; border: 1px solid #e2e8f0;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('v_{height}'), {{backgroundColor:'white'}});
        viewer.addModel(`{r_esc}`, "pdb");
        viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', opacity:0.8}}}});
        viewer.addModel(`{l_esc}`, "pdb");
        viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: '{color}', radius: 0.3}}}});
        viewer.zoomTo(); viewer.render();
    </script>
    """
    st.components.v1.html(html, height=height+15)

def extract_first_pose(pdbqt_file):
    pose = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            pose.append(line)
            if line.startswith("ENDMDL"): break
    return "".join(pose)

# ==================== MAIN UI ====================
st.markdown("<div style='text-align:center; background:linear-gradient(135deg, #1e3a8a, #3b82f6); color:white; padding:20px; border-radius:8px; margin-bottom:20px;'><h1>🌿 PhotoDock IKS / DravyaDock</h1><h3>Real Computational Photopharmacology</h3></div>", unsafe_allow_html=True)

if not vina_ready:
    st.error("🚨 AutoDock Vina binary failed to download or mount. Ensure network access.")
    st.stop()

# ----------------- STEP 1: SETUP -----------------
if st.session_state.current_step == 1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("1. Dravyaguna Phytochemical Profile")
        lig_key = st.selectbox("Select Ayurvedic Photosensitizer:", list(IKS_LIGANDS.keys()))
        lig_data = IKS_LIGANDS[lig_key]
        
        st.markdown(f"""
        <div class="ayurveda-card">
            <h3 style="margin-top:0; color:#1b5e20;">{lig_key}</h3>
            <div class="shloka">{lig_data['shloka']}</div>
            <p><b>Dravyaguna:</b> {lig_data['dravyaguna']}</p>
            <p><b>Classical Action:</b> {lig_data['karma']}</p>
            <hr>
            <p><b>Target Receptor:</b> {lig_data['target']} <i>(Auto-selected)</i></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("2. Ligand Topological Topology (Dark State)")
        b64_img = get_2d_image_base64(lig_data['smiles_dark'])
        st.markdown(f'<div style="text-align:center; background:#fff; padding:10px; border-radius:8px; border:1px solid #ddd;"><img src="data:image/png;base64,{b64_img}" style="max-width:100%;"></div>', unsafe_allow_html=True)
        st.code(lig_data['smiles_dark'], language="text")

    st.write("---")
    if st.button("🚀 Prepare Structures & Execute Ground State Docking", type="primary", use_container_width=True):
        st.session_state.selected_ligand = lig_key
        st.session_state.selected_receptor = lig_data['target']
        rec_data = RECEPTORS[lig_data['target']]
        
        with st.spinner("Downloading RCSB Target & Generating 3D Coordinates..."):
            convert_pdb_to_pdbqt(fetch_pdb_from_rcsb(rec_data['pdb_id']), "protein.pdbqt")
            convert_smiles_to_pdbqt(lig_data['smiles_dark'], "ligand_dark.pdbqt")
            
        with st.spinner("Running AutoDock Vina Optimization (Dark State)..."):
            cmd = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_dark.pdbqt",
                   "--center_x", str(rec_data['grid_center'][0]), "--center_y", str(rec_data['grid_center'][1]), "--center_z", str(rec_data['grid_center'][2]),
                   "--size_x", str(rec_data['grid_size'][0]), "--size_y", str(rec_data['grid_size'][1]), "--size_z", str(rec_data['grid_size'][2]),
                   "--exhaustiveness", "8", "--out", "dock_dark.pdbqt"]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # --- CIRCUIT BREAKER ---
            if not os.path.exists("dock_dark.pdbqt"):
                st.error("🚨 AutoDock Vina crashed during Dark State execution. Error Log:")
                st.code(process.stderr or process.stdout)
                st.stop()
                
            match = re.search(r"^\s*1\s+([-+]?\d+\.\d+)", process.stdout, re.MULTILINE)
            st.session_state.dark_affinity = float(match.group(1)) if match else 0.0
            st.session_state.dark_poses = "dock_dark.pdbqt"
            
        with st.spinner("Executing Universal Force Field (UFF) Relaxation..."):
            pre, post = execute_uff_complex_minimization("protein.pdbqt", extract_first_pose("dock_dark.pdbqt"))
            st.session_state.dark_uff = {"pre": pre, "post": post, "delta": round(float(post)-float(pre), 2) if post != "N/A" else "N/A"}
            
        st.session_state.current_step = 2
        st.rerun()

# ----------------- STEP 2: DARK STATE & ISOMERIZATION -----------------
elif st.session_state.current_step == 2:
    st.subheader("🌑 Ground State Docking Complete")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>Binding Affinity</h4>
            <div class="metric-value" style="color:#1e3a8a;">{st.session_state.dark_affinity} <span style="font-size:16px;">kcal/mol</span></div>
            <hr>
            <h4>UFF Steric Relaxation</h4>
            <p>Initial: {st.session_state.dark_uff['pre']} kcal/mol | Final: {st.session_state.dark_uff['post']} kcal/mol</p>
            <p style="color:#c62828; font-weight:bold;">Δ Energy: {st.session_state.dark_uff['delta']} kcal/mol</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='shloka'>सूर्यसंयोग: (Photoactivation)</div>", unsafe_allow_html=True)
        st.info("Applying UV photons triggers a structural isomerization (e.g., *trans* to *cis*, or photo-reduction), altering the topological fit within the receptor pocket.")
        
        if st.button("💥 Fire Photons (Isomerize & Re-Dock)", type="primary", use_container_width=True):
            lig_data = IKS_LIGANDS[st.session_state.selected_ligand]
            rec_data = RECEPTORS[st.session_state.selected_receptor]
            
            with st.spinner("Generating photo-adduct / isomer topology..."):
                convert_smiles_to_pdbqt(lig_data['smiles_light'], "ligand_light.pdbqt")
                
            with st.spinner("Running AutoDock Vina (Activated Light State)..."):
                cmd = ["./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand_light.pdbqt",
                       "--center_x", str(rec_data['grid_center'][0]), "--center_y", str(rec_data['grid_center'][1]), "--center_z", str(rec_data['grid_center'][2]),
                       "--size_x", str(rec_data['grid_size'][0]), "--size_y", str(rec_data['grid_size'][1]), "--size_z", str(rec_data['grid_size'][2]),
                       "--exhaustiveness", "8", "--out", "dock_light.pdbqt"]
                       
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                # --- CIRCUIT BREAKER ---
                if not os.path.exists("dock_light.pdbqt"):
                    st.error("🚨 AutoDock Vina crashed during Light State execution. Error Log:")
                    st.code(process.stderr or process.stdout)
                    st.stop()
                    
                match = re.search(r"^\s*1\s+([-+]?\d+\.\d+)", process.stdout, re.MULTILINE)
                st.session_state.light_affinity = float(match.group(1)) if match else 0.0
                st.session_state.light_poses = "dock_light.pdbqt"
                
            with st.spinner("Executing UFF Minimization..."):
                pre, post = execute_uff_complex_minimization("protein.pdbqt", extract_first_pose("dock_light.pdbqt"))
                st.session_state.light_uff = {"pre": pre, "post": post, "delta": round(float(post)-float(pre), 2) if post != "N/A" else "N/A"}
                
            st.session_state.current_step = 3
            st.rerun()

    with col2:
        st.markdown("**Interactive 3D Complex (Dark State)**")
        render_3d_viewer("protein.pdbqt", extract_first_pose(st.session_state.dark_poses), height=450)

# ----------------- STEP 3: COMPARISON -----------------
elif st.session_state.current_step == 3:
    st.subheader("☀️ Photopharmacology Analysis Complete")
    
    d_aff = st.session_state.dark_affinity
    l_aff = st.session_state.light_affinity
    delta = round(d_aff - l_aff, 2)
    fold = 10 ** (delta / 1.36) if delta > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card' style='text-align:center;'><h4>🌑 Ground State (Inactive)</h4><div class='metric-value'>{d_aff}</div><p>kcal/mol</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card' style='text-align:center; border-left: 5px solid #fbc02d;'><h4>☀️ Light State (Activated)</h4><div class='metric-value' style='color:#f57f17;'>{l_aff}</div><p>kcal/mol</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card' style='text-align:center; background:#e8f5e9;'><h4>📊 Selectivity (ΔΔG)</h4><div class='metric-value' style='color:#2e7d32;'>{delta}</div><p>~{fold:.1f}x Fold Change in MIC</p></div>", unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("#### 🌑 Systemic Ground State Complex")
        render_3d_viewer("protein.pdbqt", extract_first_pose(st.session_state.dark_poses), height=400, color="greenCarbon")
    with v2:
        st.markdown("#### ☀️ Locally Photoactivated Complex")
        render_3d_viewer("protein.pdbqt", extract_first_pose(st.session_state.light_poses), height=400, color="orangeCarbon")

    st.write("---")
    if st.button("🔄 Start New Experiment", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
