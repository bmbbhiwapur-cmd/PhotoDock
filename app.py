# app.py - PhotoDock IKS Main Application

import streamlit as st
import time

from modules.ligand_library import IKS_LIGANDS
from modules.receptor_library import RECEPTORS
from modules.azolog_builder import generate_trans_cis_pair, get_molecular_properties
from modules.docking_engine import run_simulated_docking
from modules.residue_analyzer import analyze_residues
from modules.isomerizer import PHOTOSWITCH_DATA, isomerize
from modules.visualizer import show_placeholder_3d, show_isomer_animation
from modules.utils import calculate_selectivity

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PhotoDock IKS | Light-Activated Antimicrobial Docking",
    page_icon="💡",
    layout="wide",
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .card-dark {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .card-light {
        background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .card-green {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .energy-big {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    .residue-chip {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 8px 15px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.9em;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
    }
    .step {
        flex: 1;
        text-align: center;
        padding: 10px;
        background: #f0f0f0;
        border-radius: 10px;
        margin: 0 5px;
        font-size: 0.8em;
    }
    .step.active {
        background: #667eea;
        color: white;
        font-weight: bold;
    }
    .step.completed {
        background: #27ae60;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 1
if 'selected_ligand' not in st.session_state:
    st.session_state.selected_ligand = None
if 'selected_receptor' not in st.session_state:
    st.session_state.selected_receptor = None
if 'current_isomer' not in st.session_state:
    st.session_state.current_isomer = 'trans'
if 'dark_result' not in st.session_state:
    st.session_state.dark_result = None
if 'light_result' not in st.session_state:
    st.session_state.light_result = None
if 'switching_done' not in st.session_state:
    st.session_state.switching_done = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("💡", width=60)
    st.title("PhotoDock IKS")
    st.markdown("---")
    
    st.markdown("### 📜 IKS Wisdom")
    st.info(
        "**Sūrya-saṃyoga-cikitsā**\n\n"
        "Ancient Āyurvedic practice of combining plant photosensitizers "
        "with sunlight to treat infections locally. This tool brings that "
        "wisdom into modern computational drug discovery."
    )
    
    st.markdown("---")
    
    # Step indicators
    st.markdown("### Workflow Progress")
    for i in range(1, 5):
        label = {1: "Select", 2: "Dark Dock", 3: "Switch", 4: "Light Dock"}[i]
        if i < st.session_state.current_phase:
            st.success(f"✅ Phase {i}: {label}")
        elif i == st.session_state.current_phase:
            st.info(f"🔄 Phase {i}: {label}")
        else:
            st.markdown(f"⏳ Phase {i}: {label}")
    
    st.markdown("---")
    st.caption("PhotoDock IKS v1.0 | Educational Demo")
    st.caption("Built with Streamlit • RDKit • ❤️")

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>💡 PhotoDock IKS</h1>
    <h3>From Ancient Sun Therapy to Red-Light Precision Antibiotics</h3>
    <p>Select an IKS natural photosensitizer, dock it to a bacterial target, 
    switch it with light, and watch the binding change.</p>
</div>
""", unsafe_allow_html=True)

# ==================== PHASE 1: SELECTION ====================
if st.session_state.current_phase == 1:
    st.markdown("## Phase 1: Select Ligand & Receptor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌿 Natural Photosensitizer (IKS)")
        
        ligand_options = list(IKS_LIGANDS.keys())
        ligand_display = [f"{l.split('(')[0].strip()} — {l.split('(')[1].replace(')','')}" for l in ligand_options]
        
        selected_display = st.selectbox(
            "Select Phytochemical",
            ligand_display,
            help="Choose a natural photosensitizer from Indian medicinal plants"
        )
        
        if selected_display:
            idx = ligand_display.index(selected_display)
            ligand_name = ligand_options[idx]
            st.session_state.selected_ligand = ligand_name
            
            lig = IKS_LIGANDS[ligand_name]
            
            st.markdown(f"""
            <div style="background:#e8f5e9; padding:20px; border-radius:15px; border-left:5px solid #2e7d32;">
                <h4>🌱 {lig['source_plant']}</h4>
                <p><b>📜 Traditional Use:</b> {lig['iks_use']}</p>
                <p><b>🎯 Target Class:</b> {lig['target_class']}</p>
                <p><b>⚙️ Mechanism:</b> {lig['mechanism']}</p>
                <p><b>MW:</b> {lig['molecular_weight']} g/mol | <b>logP:</b> {lig['logP']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**SMILES Notation:**")
            st.code(lig['smiles'])
    
    with col2:
        st.markdown("### 🦠 Bacterial Receptor")
        
        receptor_options = list(RECEPTORS.keys())
        selected_receptor = st.selectbox(
            "Select Target Protein",
            receptor_options,
            help="Choose a bacterial protein target for docking"
        )
        st.session_state.selected_receptor = selected_receptor
        
        if selected_receptor:
            rec = RECEPTORS[selected_receptor]
            
            st.markdown(f"""
            <div style="background:#fce4ec; padding:20px; border-radius:15px; border-left:5px solid #c62828;">
                <h4>📊 PDB ID: {rec['pdb_id']}</h4>
                <p><b>🧫 Organism:</b> {rec['organism']}</p>
                <p><b>📝 Function:</b> {rec['description']}</p>
                <p><b>🔑 Key Residues:</b></p>
                <p>{' '.join([f'<span class="residue-chip">{r}</span>' for r in rec['key_residues']])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🚀 Start Dark-State Docking", type="primary", use_container_width=True):
        if st.session_state.selected_ligand and st.session_state.selected_receptor:
            st.session_state.current_phase = 2
            st.rerun()
        else:
            st.error("Please select both a ligand and a receptor.")

# ==================== PHASE 2: DARK-STATE DOCKING ====================
elif st.session_state.current_phase == 2:
    st.markdown("## Phase 2: Dark-State Docking (Trans Isomer)")
    st.info("🌑 The molecule is in its **trans** (thermally stable) form — the state before light activation.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔬 Run Docking Simulation", type="primary"):
            with st.spinner("Running AutoDock Vina simulation..."):
                progress = st.progress(0)
                for i in range(10):
                    time.sleep(0.15)
                    progress.progress((i + 1) * 10)
                
                result = run_simulated_docking(
                    st.session_state.selected_ligand,
                    st.session_state.selected_receptor,
                    'trans'
                )
                st.session_state.dark_result = result
                st.rerun()
        
        if st.session_state.dark_result:
            r = st.session_state.dark_result
            
            # Result card
            st.markdown(f"""
            <div class="card-dark">
                <h3>🌑 Dark-State Binding Energy</h3>
                <div class="energy-big">{r['best_energy']} kcal/mol</div>
                <p style="text-align:center;">Binding Affinity (Trans Isomer)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Residue interactions
            st.markdown("### 🔑 Key Interacting Residues")
            residues = analyze_residues(
                st.session_state.selected_ligand,
                st.session_state.selected_receptor,
                'trans'
            )
            
            for res in residues:
                st.markdown(f"""
                <div style="background:#263238; padding:15px; border-radius:10px; margin:5px 0; color:white;">
                    <span style="background:#546e7a; padding:5px 12px; border-radius:15px; margin-right:10px;">{res['residue']}</span>
                    <b>{res['interaction']}</b> — {res['distance']}
                    <br><small style="color:#b0bec5;">{res['role']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # All poses
            st.markdown("### All Docking Poses")
            for i, e in enumerate(r['all_modes'][:5]):
                st.markdown(f"**Pose {i+1}:** {e} kcal/mol")
    
    with col2:
        show_placeholder_3d(
            st.session_state.selected_ligand,
            st.session_state.selected_receptor,
            'trans'
        )
    
    if st.session_state.dark_result:
        st.markdown("---")
        if st.button("💡 Proceed to Photo-Isomerization", type="primary", use_container_width=True):
            st.session_state.current_phase = 3
            st.rerun()

# ==================== PHASE 3: PHOTO-SWITCHING ====================
elif st.session_state.current_phase == 3:
    st.markdown("## Phase 3: Photo-Isomerization (Light Switch)")
    st.info("💡 Shine light on the molecule to switch its shape. This is the computational equivalent of **Sūrya-saṃyoga-cikitsā**.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Select Illumination Wavelength")
        
        wavelength = st.selectbox(
            "Light Source",
            list(PHOTOSWITCH_DATA.keys()),
            help="Different wavelengths drive different isomerization directions"
        )
        
        sw = PHOTOSWITCH_DATA[wavelength]
        
        st.markdown(f"""
        <div style="background:#fff3cd; padding:15px; border-radius:10px; border:1px solid #ffc107;">
            <b>⚡ Photon Energy:</b> {sw['energy_eV']} eV<br>
            <b>🔄 Drives:</b> {sw['direction'].replace('_', ' → ')}<br>
            <b>📊 Quantum Yield:</b> {sw['quantum_yield']}<br>
            <b>📝 {sw['description']}</b>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Current State:** `{st.session_state.current_isomer.upper()}`")
        
        if st.button("💥 Fire Photons!", type="primary", use_container_width=True):
            with st.spinner("Absorbing photons... isomerizing..."):
                time.sleep(1)
                new_state = isomerize(st.session_state.current_isomer, wavelength)
                
                if new_state != st.session_state.current_isomer:
                    st.session_state.switching_done = True
                    old_state = st.session_state.current_isomer
                    st.session_state.current_isomer = new_state
                    st.rerun()
                else:
                    st.warning("No switch occurred. Try a different wavelength.")
        
        if st.session_state.switching_done:
            st.balloons()
            st.success(f"✅ Photoisomerization successful! Molecule is now **{st.session_state.current_isomer.upper()}**")
    
    with col2:
        show_placeholder_3d(
            st.session_state.selected_ligand,
            st.session_state.selected_receptor,
            st.session_state.current_isomer
        )
        
        if st.session_state.switching_done:
            show_isomer_animation('trans', st.session_state.current_isomer)
    
    if st.session_state.switching_done:
        st.markdown("---")
        if st.button("🔬 Run Light-State Docking", type="primary", use_container_width=True):
            st.session_state.current_phase = 4
            st.rerun()

# ==================== PHASE 4: LIGHT-STATE DOCKING ====================
elif st.session_state.current_phase == 4:
    st.markdown("## Phase 4: Light-State Docking & Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 Run Light-State Docking", type="primary"):
            with st.spinner("Docking the photoswitched isomer..."):
                time.sleep(1.5)
                result = run_simulated_docking(
                    st.session_state.selected_ligand,
                    st.session_state.selected_receptor,
                    st.session_state.current_isomer
                )
                st.session_state.light_result = result
                st.rerun()
        
        if st.session_state.light_result and st.session_state.dark_result:
            dark = st.session_state.dark_result['best_energy']
            light = st.session_state.light_result['best_energy']
            delta = dark - light
            fold = calculate_selectivity(dark, light)
            
            # Side-by-side comparison
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="card-dark">
                    <h4>🌑 Dark (Trans)</h4>
                    <div style="font-size:2em; font-weight:bold; text-align:center;">{dark}</div>
                    <p style="text-align:center;">kcal/mol</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="card-light">
                    <h4>☀️ Light ({st.session_state.current_isomer.upper()})</h4>
                    <div style="font-size:2em; font-weight:bold; text-align:center;">{light}</div>
                    <p style="text-align:center;">kcal/mol</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Delta and prediction
            if delta > 2:
                color = "27ae60"
                verdict = "✅ EXCELLENT photo-switch candidate!"
                detail = f"Predicted >{fold}x MIC selectivity"
            elif delta > 1:
                color = "f39c12"
                verdict = "⚠️ MODERATE switch — optimization needed"
                detail = f"Predicted ~{fold}x MIC selectivity"
            else:
                color = "c0392b"
                verdict = "❌ Poor switch — redesign recommended"
                detail = f"Only ~{fold}x selectivity"
            
            st.markdown(f"""
            <div style="background:#{color}; padding:20px; border-radius:15px; color:white; text-align:center;">
                <h3>{verdict}</h3>
                <p style="font-size:1.2em;">ΔΔG = {delta:.1f} kcal/mol | {detail}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Residues in light state
            st.markdown("### 🔑 Light-State Residue Interactions")
            residues = analyze_residues(
                st.session_state.selected_ligand,
                st.session_state.selected_receptor,
                st.session_state.current_isomer
            )
            for res in residues:
                st.markdown(f"""
                <div style="background:#f5f5f5; padding:12px; border-radius:10px; margin:5px 0;">
                    <span class="residue-chip">{res['residue']}</span>
                    <b>{res['interaction']}</b> — {res['distance']}
                    <br><small>{res['role']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        show_placeholder_3d(
            st.session_state.selected_ligand,
            st.session_state.selected_receptor,
            st.session_state.current_isomer
        )
        
        if st.session_state.light_result:
            st.markdown(f"""
            <div class="card-green">
                <h4>📊 IKS Therapeutic Prediction</h4>
                <p>Just as <b>{st.session_state.selected_ligand.split('(')[0].strip()}</b> from <b>{IKS_LIGANDS[st.session_state.selected_ligand]['source_plant']}</b> was traditionally used with sunlight for localized antimicrobial action, this photoswitchable derivative achieves the same principle with molecular precision.</p>
                <p><b>Dark (Systemic):</b> Inactive — spares gut microbiome</p>
                <p><b>Light (Wound):</b> Active — kills pathogens on-demand</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Reset button
    if st.session_state.light_result:
        st.markdown("---")
        if st.button("🔄 Start New Docking Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; padding:20px;">
    <p>💡 <b>PhotoDock IKS</b> v1.0 | Educational Demonstration</p>
    <p>Inspired by the Indian Knowledge System's <i>Sūrya-saṃyoga-cikitsā</i></p>
    <p style="font-size:0.8em;">Built with Streamlit • RDKit • Python</p>
</div>
""", unsafe_allow_html=True)
