# modules/visualizer.py

import streamlit as st

def show_placeholder_3d(ligand_name, receptor_name, isomer_state):
    """
    Display a placeholder for 3D visualization.
    In production, integrate py3Dmol for actual 3D rendering.
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        color: white;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    ">
        <h3>🧬 3D Molecular Viewer</h3>
        <p><b>Ligand:</b> {ligand_name}</p>
        <p><b>Receptor:</b> {receptor_name}</p>
        <p><b>State:</b> <span style="color: {'#f39c12' if isomer_state == 'cis' else '#3498db'}; font-weight: bold;">{isomer_state.upper()}</span></p>
        <p style="color: #888; font-size: 0.9em; margin-top: 20px;">
            [3Dmol.js rendering will appear here in full version]
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_isomer_animation(current_state, new_state):
    """Show an ASCII animation of the isomerization."""
    if current_state == "trans" and new_state == "cis":
        animation = """
        <div style="font-family: monospace; background: #000; color: #0f0; padding: 20px; border-radius: 10px;">
        <pre>
        TRANS (Elongated)          →    CIS (Bent)
        
        [Ring]-N=N-[Ring]              [Ring]    [Ring]
             |      |                      \\    /
             |      |                        N=N
             |      |                         
        Photon absorbed (hν)         Shape change complete!
        </pre>
        </div>
        """
        st.markdown(animation, unsafe_allow_html=True)
