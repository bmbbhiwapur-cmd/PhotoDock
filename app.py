# app.py - PhotoDock IKS Main Application
# Author: Sarang Dhote, Assistant Professor, Department of Chemistry
# A computational platform integrating Indian Knowledge System with modern photopharmacology

import streamlit as st
import time
import os
import pandas as pd

# Local module imports
from modules.ligand_library import IKS_LIGANDS
from modules.receptor_library import RECEPTORS
from modules.azolog_builder import generate_trans_cis_pair, get_molecular_properties
from modules.docking_engine import run_simulated_docking
from modules.residue_analyzer import analyze_residues
from modules.isomerizer import PHOTOSWITCH_DATA, isomerize
from modules.visualizer import show_placeholder_3d, show_isomer_animation
from modules.utils import calculate_selectivity, ensure_temp_dir

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="PhotoDock IKS | Light-Activated Antimicrobial Docking",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .main-header h1 { font-size: 2.5em; margin-bottom: 10px; }
    .main-header h3 { font-weight: 300; opacity: 0.9; }
    
    .card-dark {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px; padding: 20px; color: white;
        margin: 10px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .card-light {
        background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
        border-radius: 15px; padding: 20px; color: white;
        margin: 10px 0; box-shadow: 0 4px 10px rgba(231, 76, 60, 0.3);
    }
    .card-green {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        border-radius: 15px; padding: 20px; color: white;
        margin: 10px 0; box-shadow: 0 4px 10px rgba(39, 174, 96, 0.3);
    }
    .energy-big {
        font-size: 3em; font-weight: bold; text-align: center; margin: 10px 0;
    }
    .residue-chip {
        display: inline-block; background: rgba(255,255,255,0.15);
        padding: 8px 15px; border-radius: 20px; margin: 4px;
        font-size: 0.9em; border: 1px solid rgba(255,255,255,0.3); color: white;
    }
    .section-title {
        font-size: 1.2em; font-weight: 600; margin: 15px 0 10px 0;
        border-bottom: 2px solid #667eea; padding-bottom: 5px; color: #2c3e50;
    }
    .info-box {
        background: #f0f8ff; padding: 15px; border-radius: 10px;
        border-left: 4px solid #228B22; margin: 10px 0;
    }
    .info-box-red {
        background: #fff0f5; padding: 15px; border-radius: 10px;
        border-left: 4px solid #DC143C; margin: 10px 0;
    }
    .footer {
        text-align: center; color: #888; padding: 20px;
        margin-top: 30px; border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.current_phase = 1
    st.session_state.selected_ligand = None
    st.session_state.selected_receptor = None
    st.session_state.current_isomer = 'trans'
    st.session_state.dark_result = None
    st.session_state.light_result = None
    st.session_state.switching_done = False
    st.session_state.dark_docking_run = False
    st.session_state.light_docking_run = False
    st.session_state.error_message = None

# Ensure temp directory exists
ensure_temp_dir()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        "<h1 style='text-align: center; font-size: 60px; margin: 0;'>💡</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align: center; color: #667eea;'>PhotoDock IKS</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    st.markdown("### 📜 IKS Wisdom")
    st.info(
        "**Sūrya-saṃyoga-cikitsā**\n\n"
        "Ancient Āyurvedic practice of combining plant photosensitizers "
        "with sunlight to treat infections locally. This tool brings that "
        "wisdom into modern computational drug discovery."
    )
    
    st.markdown("---")
    
    st.markdown("### 🔄 Workflow Progress")
    
    phases = {
        1: "Select Ligand & Receptor",
        2: "Dark-State Docking",
        3: "Photo-Isomerization",
        4: "Light-State Docking & Results"
    }
    
    for phase_num, phase_label in phases.items():
        if phase_num < st.session_state.current_phase:
            st.success(f"✅ Phase {phase_num}: {phase_label}")
        elif phase_num == st.session_state.current_phase:
            st.info(f"🔄 Phase {phase_num}: {phase_label}")
        else:
            st.markdown(f"⏳ Phase {phase_num}: {phase_label}")
    
    st.markdown("---")
    
    if st.session_state.selected_ligand:
        st.markdown(f"**🌿 Ligand:** {st.session_state.selected_ligand.split('(')[0].strip()}")
    if st.session_state.selected_receptor:
        st.markdown(f"**🦠 Receptor:** {st.session_state.selected_receptor.split('(')[0].strip()}")
    if st.session_state.dark_result:
        st.markdown(f"**🌑 Dark Binding:** {st.session_state.dark_result['best_energy']} kcal/mol")
    if st.session_state.light_result:
        st.markdown(f"**☀️ Light Binding:** {st.session_state.light_result['best_energy']} kcal/mol")
    
    st.markdown("---")
    st.caption("PhotoDock IKS v1.0 | Educational Demo")
    st.caption("Built with Streamlit • RDKit • ❤️")
    st.caption("© Sarang Dhote, Dept. of Chemistry")

# ==================== ERROR DISPLAY ====================
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    if st.button("Clear Error"):
        st.session_state.error_message = None
        st.rerun()

# ==================== MAIN HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>💡 PhotoDock IKS</h1>
    <h3>From Ancient Sun Therapy to Red-Light Precision Antibiotics</h3>
    <p>Select an IKS natural photosensitizer, dock it to a bacterial target, 
    switch it with light, and watch the binding change — all in your browser.</p>
</div>
""", unsafe_allow_html=True)

# ==================== PHASE 1: SELECTION ====================
if st.session_state.current_phase == 1:
    st.markdown("## 🔬 Phase 1: Select Ligand & Receptor")
    st.markdown("Choose a natural photosensitizer from the Indian Knowledge System and a bacterial protein target.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-title">🌿 Natural Photosensitizer (IKS)</p>', unsafe_allow_html=True)
        
        ligand_options = list(IKS_LIGANDS.keys())
        ligand_display_names = [
            f"{l.split('(')[0].strip()} — {l.split('(')[1].replace(')','')}" 
            for l in ligand_options
        ]
        
        selected_display = st.selectbox(
            "Select Phytochemical",
            ligand_display_names,
            help="Choose a natural photosensitizer from Indian medicinal plants documented in classical Ayurvedic texts",
            key="ligand_select"
        )
        
        if selected_display:
            idx = ligand_display_names.index(selected_display)
            ligand_name = ligand_options[idx]
            st.session_state.selected_ligand = ligand_name
            
            lig = IKS_LIGANDS[ligand_name]
            
            st.markdown(f"""
            <div class="info-box">
                <h4>🌱 <b>{lig['source_plant']}</b></h4>
                <p><b>📜 Traditional IKS Use:</b> {lig['iks_use']}</p>
                <p><b>🎯 Modern Target Class:</b> {lig['target_class']}</p>
                <p><b>⚙️ Mechanism of Action:</b> {lig['mechanism']}</p>
                <p><b>⚖️ Molecular Weight:</b> {lig['molecular_weight']} g/mol | <b>💧 logP:</b> {lig['logP']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**🧪 SMILES Notation:**")
            st.code(lig['smiles'], language='text')
            
            try:
                props = get_molecular_properties(lig['smiles'])
                if props:
                    st.markdown("**📊 Computed Properties:**")
                    prop_cols = st.columns(4)
                    with prop_cols[0]:
                        st.metric("MW", f"{props.get('molecular_weight', 'N/A')}")
                    with prop_cols[1]:
                        st.metric("logP", f"{props.get('logP', 'N/A')}")
                    with prop_cols[2]:
                        st.metric("H-Donors", f"{props.get('h_bond_donors', 'N/A')}")
                    with prop_cols[3]:
                        st.metric("H-Acceptors", f"{props.get('h_bond_acceptors', 'N/A')}")
            except Exception:
                pass
    
    with col2:
        st.markdown('<p class="section-title">🦠 Bacterial Target Receptor</p>', unsafe_allow_html=True)
        
        receptor_options = list(RECEPTORS.keys())
        selected_receptor = st.selectbox(
            "Select Target Protein",
            receptor_options,
            help="Choose a bacterial protein target for molecular docking",
            key="receptor_select"
        )
        st.session_state.selected_receptor = selected_receptor
        
        if selected_receptor:
            rec = RECEPTORS[selected_receptor]
            
            st.markdown(f"""
            <div class="info-box-red">
                <h4>📊 PDB ID: <b>{rec['pdb_id']}</b></h4>
                <p><b>🧫 Organism:</b> {rec['organism']}</p>
                <p><b>📝 Biological Function:</b> {rec['description']}</p>
                <p><b>📍 Binding Site:</b> {rec['binding_site']}</p>
                <p><b>🔑 Key Active Site Residues:</b></p>
                <p>{' '.join([f'<span class="residue-chip" style="background:#667eea; color:white;">{r}</span>' for r in rec['key_residues']])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            target_reasons = {
                "DNA Gyrase (S. aureus)": "Essential for bacterial DNA replication. Target of fluoroquinolones. Plumbagin is a known natural inhibitor.",
                "FtsZ (S. aureus)": "Key cell division protein. No human homolog — excellent selectivity potential.",
                "DHFR (E. coli)": "Folate pathway enzyme. Validated trimethoprim target. Azo-derivatives can photo-control activity.",
                "PBP2a (MRSA)": "Confers methicillin resistance. New photoswitchable inhibitors could restore β-lactam efficacy.",
            }
            st.info(target_reasons.get(selected_receptor, "Validated antibacterial drug target."))
    
    st.markdown("---")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button("🚀 Start Dark-State Docking", type="primary", use_container_width=True):
            if st.session_state.selected_ligand and st.session_state.selected_receptor:
                st.session_state.current_phase = 2
                st.session_state.dark_docking_run = False
                st.rerun()
            else:
                st.session_state.error_message = "⚠️ Please select both a ligand and a receptor before proceeding."
                st.rerun()

# ==================== PHASE 2: DARK-STATE DOCKING ====================
elif st.session_state.current_phase == 2:
    st.markdown("## 🌑 Phase 2: Dark-State Docking (Trans Isomer)")
    st.info(
        "The molecule is in its **trans** (thermally stable, elongated) form. "
        "This represents the 'inactive' state before light activation, analogous to "
        "the herbal preparation before sun exposure in traditional Sūrya-saṃyoga-cikitsā."
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if not st.session_state.dark_docking_run:
            if st.button("🔬 Run Dark-State Docking Simulation", type="primary", use_container_width=True):
                with st.spinner("🧬 Preparing ligand... Generating 3D coordinates..."):
                    time.sleep(0.5)
                
                # FIXED: Progress bar values MUST be between 0.0 and 1.0
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                steps = [
                    "Generating trans isomer geometry...",
                    "Preparing PDBQT files...",
                    "Setting up grid box around active site...",
                    "Running AutoDock Vina scoring function...",
                    "Evaluating binding poses...",
                    "Calculating interaction energies...",
                    "Analyzing residue contacts...",
                    "Finalizing results..."
                ]
                
                total_steps = len(steps)
                for i, step in enumerate(steps):
                    status_text.text(f"⏳ {step}")
                    time.sleep(0.2)
                    # FIX: Divide by total_steps to keep value between 0.0 and 1.0
                    progress_value = (i + 1) / total_steps
                    progress_bar.progress(progress_value)
                
                try:
                    result = run_simulated_docking(
                        st.session_state.selected_ligand,
                        st.session_state.selected_receptor,
                        'trans'
                    )
                    st.session_state.dark_result = result
                    st.session_state.dark_docking_run = True
                    status_text.empty()
                    progress_bar.empty()
                    st.rerun()
                except Exception as e:
                    st.session_state.error_message = f"Docking error: {str(e)}"
                    st.rerun()
        
        if st.session_state.dark_result and st.session_state.dark_docking_run:
            r = st.session_state.dark_result
            
            st.toast("✅ Dark-state docking complete!", icon="🌑")
            
            st.markdown(f"""
            <div class="card-dark">
                <h3 style="text-align:center;">🌑 Dark-State Binding Affinity</h3>
                <div class="energy-big">{r['best_energy']} kcal/mol</div>
                <p style="text-align:center; opacity:0.9;">Trans Isomer — Thermally Stable Form</p>
                <p style="text-align:center; font-size:0.9em;">
                    More negative = stronger binding | Typical drug range: -6 to -12 kcal/mol
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            energy = r['best_energy']
            if energy <= -9:
                st.success(f"🟢 **Strong binder** — {energy} kcal/mol indicates excellent target affinity")
            elif energy <= -7:
                st.info(f"🟡 **Moderate binder** — {energy} kcal/mol indicates good target affinity")
            else:
                st.warning(f"🟠 **Weak binder** — {energy} kcal/mol may need structural optimization")
            
            st.markdown('<p class="section-title">🔑 Key Interacting Amino Acid Residues</p>', unsafe_allow_html=True)
            
            try:
                residues = analyze_residues(
                    st.session_state.selected_ligand,
                    st.session_state.selected_receptor,
                    'trans'
                )
                
                for res in residues:
                    if 'H-bond' in res['interaction'] or 'Hydrogen' in res['interaction']:
                        border_color = "#3498db"
                    elif 'π' in res['interaction'] or 'cation' in res['interaction']:
                        border_color = "#9b59b6"
                    else:
                        border_color = "#7f8c8d"
                    
                    st.markdown(f"""
                    <div style="background:#263238; padding:15px; border-radius:10px; margin:8px 0; color:white; border-left:4px solid {border_color};">
                        <span style="background:#546e7a; padding:5px 12px; border-radius:15px; margin-right:10px; font-weight:bold;">{res['residue']}</span>
                        <b>{res['interaction']}</b> — <span style="color:#81d4fa;">{res['distance']}</span>
                        <br><small style="color:#b0bec5;">{res['role']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.info("Residue interaction data not available for this combination.")
            
            st.markdown('<p class="section-title">📊 All Docking Poses</p>', unsafe_allow_html=True)
            poses = r['all_modes'][:5]
            pose_cols = st.columns(len(poses))
            for i, (col, energy_val) in enumerate(zip(pose_cols, poses)):
                with col:
                    st.metric(f"Pose {i+1}", f"{energy_val} kcal/mol")
    
    with col2:
        show_placeholder_3d(
            st.session_state.selected_ligand if st.session_state.selected_ligand else "Not selected",
            st.session_state.selected_receptor if st.session_state.selected_receptor else "Not selected",
            'trans'
        )
        
        if st.session_state.dark_result:
            st.markdown("---")
            st.markdown("### 📈 Docking Statistics")
            r = st.session_state.dark_result
            st.markdown(f"""
            - **Best Pose:** {r['best_energy']} kcal/mol
            - **Total Poses:** {len(r['all_modes'])}
            - **Isomer:** TRANS (elongated)
            - **Status:** Thermally stable (dark)
            """)
    
    if st.session_state.dark_docking_run:
        st.markdown("---")
        nav_col1, nav_col2, _ = st.columns([1, 2, 1])
        with nav_col2:
            if st.button("💡 Proceed to Photo-Isomerization Phase", type="primary", use_container_width=True):
                st.session_state.current_phase = 3
                st.session_state.switching_done = False
                st.session_state.current_isomer = 'trans'
                st.rerun()
        with nav_col1:
            if st.button("↩️ Back"):
                st.session_state.current_phase = 1
                st.rerun()

# ==================== PHASE 3: PHOTO-ISOMERIZATION ====================
elif st.session_state.current_phase == 3:
    st.markdown("## 💡 Phase 3: Photo-Isomerization (Light Switch)")
    st.info(
        "Shine light on the molecule to switch its shape from trans (elongated) to cis (bent). "
        "This is the computational equivalent of **Sūrya-saṃyoga** — combining the herbal preparation with sunlight."
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<p class="section-title">🎛️ Select Illumination Wavelength</p>', unsafe_allow_html=True)
        
        wavelength = st.selectbox(
            "Light Source (Wavelength)",
            list(PHOTOSWITCH_DATA.keys()),
            help="Different wavelengths drive different isomerization directions. UV → trans→cis, Green/Red → cis→trans"
        )
        
        if wavelength:
            sw = PHOTOSWITCH_DATA[wavelength]
            
            st.markdown(f"""
            <div style="background:#fff3cd; padding:20px; border-radius:10px; border:1px solid #ffc107; margin:10px 0;">
                <h4>⚡ Photon Properties</h4>
                <table style="width:100%; color:#333;">
                    <tr><td><b>Photon Energy:</b></td><td>{sw['energy_eV']} eV</td></tr>
                    <tr><td><b>Drives:</b></td><td>{sw['direction'].replace('_', ' → ')}</td></tr>
                    <tr><td><b>Quantum Yield:</b></td><td>{sw['quantum_yield']} ({int(sw['quantum_yield']*100)}% efficiency)</td></tr>
                    <tr><td><b>Clinical Relevance:</b></td><td>{sw['description']}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="text-align:center; margin:20px 0;">
                <h4>Current Molecular State:</h4>
                <h2 style="color:{'#3498db' if st.session_state.current_isomer == 'trans' else '#e74c3c'}; font-size:3em;">
                    {st.session_state.current_isomer.upper()}
                </h2>
                <p>{'Elongated, thermally stable form' if st.session_state.current_isomer == 'trans' else 'Bent, photogenerated form'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if not st.session_state.switching_done:
                _, center_col, _ = st.columns([1, 2, 1])
                with center_col:
                    if st.button("💥 Fire Photons! (Isomerize)", type="primary", use_container_width=True):
                        with st.spinner("⚡ Absorbing photons... Molecular rearrangement in progress..."):
                            time.sleep(1.0)
                            new_state = isomerize(st.session_state.current_isomer, wavelength)
                            
                            if new_state != st.session_state.current_isomer:
                                st.session_state.switching_done = True
                                st.session_state.current_isomer = new_state
                                st.session_state.light_docking_run = False
                                st.session_state.light_result = None
                                st.rerun()
                            else:
                                st.warning(
                                    f"⚠️ No isomerization occurred. "
                                    f"The molecule is already in {st.session_state.current_isomer.upper()} state. "
                                    f"Try a wavelength that drives {st.session_state.current_isomer} → "
                                    f"{'cis' if st.session_state.current_isomer == 'trans' else 'trans'}."
                                )
            
            if st.session_state.switching_done:
                st.toast("⚡ Photoisomerization successful!", icon="💡")
                st.success(f"✅ Molecule successfully switched to **{st.session_state.current_isomer.upper()}** state!")
                show_isomer_animation('trans', st.session_state.current_isomer)
    
    with col2:
        show_placeholder_3d(
            st.session_state.selected_ligand if st.session_state.selected_ligand else "Not selected",
            st.session_state.selected_receptor if st.session_state.selected_receptor else "Not selected",
            st.session_state.current_isomer
        )
        
        st.markdown("---")
        st.markdown("### 📜 IKS Connection")
        st.markdown("""
        <div style="background:#e8f5e9; padding:15px; border-radius:10px;">
            <p><b>Sūrya-saṃyoga-cikitsā</b> (Sun-combined therapy) recognized that certain 
            plant preparations become therapeutically active only when combined with sunlight.</p>
            <p>This photoswitching step computationally recreates that ancient principle — 
            the photon changes the molecule's shape, altering its biological activity.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.switching_done:
        st.markdown("---")
        nav_col1, nav_col2, _ = st.columns([1, 2, 1])
        with nav_col2:
            if st.button("🔬 Run Light-State Docking", type="primary", use_container_width=True):
                st.session_state.current_phase = 4
                st.session_state.light_docking_run = False
                st.rerun()
        with nav_col1:
            if st.button("↩️ Back"):
                st.session_state.current_phase = 2
                st.rerun()

# ==================== PHASE 4: LIGHT-STATE DOCKING ====================
elif st.session_state.current_phase == 4:
    st.markdown("## ☀️ Phase 4: Light-State Docking & Complete Results")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if not st.session_state.light_docking_run:
            if st.button("🔬 Run Light-State Docking", type="primary", use_container_width=True):
                with st.spinner("🧬 Docking the photoswitched isomer..."):
                    time.sleep(1.5)
                
                try:
                    result = run_simulated_docking(
                        st.session_state.selected_ligand,
                        st.session_state.selected_receptor,
                        st.session_state.current_isomer
                    )
                    st.session_state.light_result = result
                    st.session_state.light_docking_run = True
                    st.rerun()
                except Exception as e:
                    st.session_state.error_message = f"Light docking error: {str(e)}"
                    st.rerun()
        
        if st.session_state.light_result and st.session_state.dark_result:
            st.toast("✅ Light-state docking complete!", icon="☀️")
            
            dark = st.session_state.dark_result['best_energy']
            light = st.session_state.light_result['best_energy']
            delta = dark - light
            fold = calculate_selectivity(dark, light)
            
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown(f"""
                <div class="card-dark">
                    <h4 style="text-align:center;">🌑 Dark State</h4>
                    <p style="text-align:center;">Trans Isomer</p>
                    <div style="font-size:2.5em; font-weight:bold; text-align:center;">{dark}</div>
                    <p style="text-align:center;">kcal/mol</p>
                    <p style="text-align:center; font-size:0.8em;">Systemically inactive</p>
                </div>
                """, unsafe_allow_html=True)
            
            with comp_col2:
                st.markdown(f"""
                <div class="card-light">
                    <h4 style="text-align:center;">☀️ Light State</h4>
                    <p style="text-align:center;">{st.session_state.current_isomer.upper()} Isomer</p>
                    <div style="font-size:2.5em; font-weight:bold; text-align:center;">{light}</div>
                    <p style="text-align:center;">kcal/mol</p>
                    <p style="text-align:center; font-size:0.8em;">Locally activated</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            if delta > 3:
                verdict_color = "#27ae60"
                verdict_text = "✅ EXCELLENT Photoswitch!"
                recommendation = "This is a promising photoantibiotic candidate. Consider experimental validation."
            elif delta > 1.5:
                verdict_color = "#f39c12"
                verdict_text = "⚠️ MODERATE Photoswitch"
                recommendation = "Consider modifying the azo bridge substituents for improved switching."
            else:
                verdict_color = "#c0392b"
                verdict_text = "❌ POOR Photoswitch"
                recommendation = "Try a different ligand-receptor pair or redesign the azo insertion point."
            
            st.markdown(f"""
            <div style="background:{verdict_color}; padding:25px; border-radius:15px; color:white; text-align:center; margin:20px 0;">
                <h2>{verdict_text}</h2>
                <div style="background:rgba(255,255,255,0.2); padding:15px; border-radius:10px; margin:15px 0;">
                    <h3>ΔΔG = {delta:.1f} kcal/mol</h3>
                    <p>Predicted MIC Selectivity: <b>~{fold}-fold</b></p>
                </div>
                <p><i>{recommendation}</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<p class="section-title">🔑 Residue Interaction Comparison</p>', unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("**🌑 Dark State (Trans):**")
                try:
                    residues_dark = analyze_residues(
                        st.session_state.selected_ligand,
                        st.session_state.selected_receptor,
                        'trans'
                    )
                    for res in residues_dark:
                        st.markdown(f"""
                        <div style="background:#e8f5e9; padding:10px; border-radius:8px; margin:5px 0;">
                            <b>{res['residue']}</b>: {res['interaction']} ({res['distance']})
                        </div>
                        """, unsafe_allow_html=True)
                except Exception:
                    st.info("Data not available")
            
            with res_col2:
                st.markdown(f"**☀️ Light State ({st.session_state.current_isomer.upper()}):**")
                try:
                    residues_light = analyze_residues(
                        st.session_state.selected_ligand,
                        st.session_state.selected_receptor,
                        st.session_state.current_isomer
                    )
                    for res in residues_light:
                        st.markdown(f"""
                        <div style="background:#fce4ec; padding:10px; border-radius:8px; margin:5px 0;">
                            <b>{res['residue']}</b>: {res['interaction']} ({res['distance']})
                        </div>
                        """, unsafe_allow_html=True)
                except Exception:
                    st.info("Data not available")
    
    with col2:
        show_placeholder_3d(
            st.session_state.selected_ligand if st.session_state.selected_ligand else "Not selected",
            st.session_state.selected_receptor if st.session_state.selected_receptor else "Not selected",
            st.session_state.current_isomer
        )
        
        if st.session_state.light_result:
            st.markdown("---")
            st.markdown("### 📜 IKS Therapeutic Summary")
            
            lig_name = st.session_state.selected_ligand.split('(')[0].strip() if st.session_state.selected_ligand else ""
            plant_name = IKS_LIGANDS.get(st.session_state.selected_ligand, {}).get('source_plant', '')
            iks_use = IKS_LIGANDS.get(st.session_state.selected_ligand, {}).get('iks_use', '')
            
            st.markdown(f"""
            <div class="card-green">
                <h4>🧘 IKS-to-Modern Translation</h4>
                <p>Just as <b>{lig_name}</b> from <b>{plant_name}</b> was traditionally used 
                with sunlight for <i>{iks_use}</i>, this photoswitchable derivative achieves 
                the same principle with molecular precision.</p>
                <hr style="opacity:0.3;">
                <p><b>🌑 Dark (Systemic Circulation):</b> Inactive — spares gut microbiome</p>
                <p><b>☀️ Light (Infection Site):</b> Active — kills pathogens on-demand</p>
                <p><b>🎯 Principle:</b> Sūrya-saṃyoga-cikitsā reimagined through photopharmacology</p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.light_docking_run:
        st.markdown("---")
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        
        with nav_col1:
            if st.button("🔄 New Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'app_initialized':
                        del st.session_state[key]
                st.session_state.current_phase = 1
                st.session_state.selected_ligand = None
                st.session_state.selected_receptor = None
                st.session_state.current_isomer = 'trans'
                st.session_state.dark_result = None
                st.session_state.light_result = None
                st.session_state.switching_done = False
                st.session_state.dark_docking_run = False
                st.session_state.light_docking_run = False
                st.session_state.error_message = None
                st.rerun()
        
        with nav_col3:
            if st.session_state.light_result and st.session_state.dark_result:
                data = {
                    "Parameter": [
                        "Ligand", "Receptor", "Dark Binding (kcal/mol)",
                        "Light Binding (kcal/mol)", "ΔΔG (kcal/mol)",
                        "Predicted MIC Fold Change", "Photoswitch State"
                    ],
                    "Value": [
                        st.session_state.selected_ligand,
                        st.session_state.selected_receptor,
                        st.session_state.dark_result['best_energy'],
                        st.session_state.light_result['best_energy'],
                        round(st.session_state.dark_result['best_energy'] - st.session_state.light_result['best_energy'], 1),
                        calculate_selectivity(
                            st.session_state.dark_result['best_energy'],
                            st.session_state.light_result['best_energy']
                        ),
                        st.session_state.current_isomer.upper()
                    ]
                }
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "photodock_results.csv",
                    "text/csv",
                    key='download-csv'
                )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>💡 <b>PhotoDock IKS</b> v1.0 | Educational Demonstration Platform</p>
    <p>Inspired by the Indian Knowledge System's <i>Sūrya-saṃyoga-cikitsā</i> (Sun-Combined Therapy)</p>
    <p>Bridging ancient Āyurvedic wisdom with modern computational photopharmacology</p>
    <p style="font-size:0.8em; margin-top:10px;">
        Built with Streamlit • RDKit • Python | 
        <b>Sarang Dhote</b>, Assistant Professor, Department of Chemistry
    </p>
    <p style="font-size:0.7em; color:#aaa;">
        © 2024 | For educational and research purposes
    </p>
</div>
""", unsafe_allow_html=True)
