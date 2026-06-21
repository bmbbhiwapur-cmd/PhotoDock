# modules/docking_engine.py

import time
import random

# Simulated binding data for demonstration
# In production, this would call AutoDock Vina via subprocess
SIMULATED_BINDING = {
    "DNA Gyrase (S. aureus)": {
        "Plumbagin (Chitraka)": {"trans": -9.8, "cis": -5.1},
        "Marmelosin (Bilva)": {"trans": -8.3, "cis": -4.9},
        "Lawsone (Mendhi)": {"trans": -7.6, "cis": -3.8},
        "Psoralen (Bakuchi)": {"trans": -8.9, "cis": -4.2},
        "Bergapten (Bijapura)": {"trans": -8.1, "cis": -4.5},
    },
    "FtsZ (S. aureus)": {
        "Plumbagin (Chitraka)": {"trans": -7.9, "cis": -4.3},
        "Marmelosin (Bilva)": {"trans": -9.2, "cis": -3.9},
        "Lawsone (Mendhi)": {"trans": -7.1, "cis": -4.8},
        "Psoralen (Bakuchi)": {"trans": -7.5, "cis": -5.0},
        "Bergapten (Bijapura)": {"trans": -7.3, "cis": -5.2},
    },
    "DHFR (E. coli)": {
        "Plumbagin (Chitraka)": {"trans": -8.6, "cis": -5.5},
        "Marmelosin (Bilva)": {"trans": -7.8, "cis": -4.6},
        "Lawsone (Mendhi)": {"trans": -7.4, "cis": -4.1},
        "Psoralen (Bakuchi)": {"trans": -8.2, "cis": -4.9},
        "Bergapten (Bijapura)": {"trans": -7.7, "cis": -4.4},
    },
    "PBP2a (MRSA)": {
        "Plumbagin (Chitraka)": {"trans": -8.9, "cis": -4.7},
        "Marmelosin (Bilva)": {"trans": -8.1, "cis": -4.4},
        "Lawsone (Mendhi)": {"trans": -7.2, "cis": -3.9},
        "Psoralen (Bakuchi)": {"trans": -8.5, "cis": -4.1},
        "Bergapten (Bijapura)": {"trans": -7.9, "cis": -4.3},
    },
}

def run_simulated_docking(ligand_name, receptor_name, isomer_state):
    """
    Simulated docking returning binding energy.
    Replace with real Vina calls in production.
    """
    # Simulate computation time
    time.sleep(1.5)
    
    # Get base binding energy
    base_energy = SIMULATED_BINDING.get(receptor_name, {}).get(ligand_name, {}).get(isomer_state, -6.5)
    
    # Add small random variation
    variation = random.uniform(-0.3, 0.3)
    best_energy = round(base_energy + variation, 1)
    
    # Generate multiple poses
    all_modes = [best_energy]
    for i in range(1, 5):
        all_modes.append(round(best_energy + i * 0.8 + random.uniform(-0.2, 0.2), 1))
    
    return {
        "best_energy": best_energy,
        "all_modes": all_modes,
        "isomer_state": isomer_state,
        "ligand": ligand_name,
        "receptor": receptor_name,
    }
