# modules/isomerizer.py

import random

PHOTOSWITCH_DATA = {
    "UV (365 nm)": {
        "direction": "trans_to_cis",
        "quantum_yield": 0.45,
        "energy_eV": 3.40,
        "description": "Ultraviolet light. Used in classical PUVA therapy.",
    },
    "Violet (405 nm)": {
        "direction": "trans_to_cis",
        "quantum_yield": 0.38,
        "energy_eV": 3.06,
        "description": "Violet visible light. Gentler than UV, still drives trans→cis.",
    },
    "Green (530 nm)": {
        "direction": "cis_to_trans",
        "quantum_yield": 0.52,
        "energy_eV": 2.34,
        "description": "Green light. Reverses the switch. Clinically safe.",
    },
    "Red (630 nm)": {
        "direction": "cis_to_trans",
        "quantum_yield": 0.41,
        "energy_eV": 1.97,
        "description": "Red light. Deep tissue penetration. Therapeutic window.",
    },
    "NIR (750 nm)": {
        "direction": "cis_to_trans",
        "quantum_yield": 0.28,
        "energy_eV": 1.65,
        "description": "Near-infrared. Maximum tissue penetration, no photodamage.",
    },
}

def isomerize(current_state, wavelength):
    """
    Simulate photoisomerization.
    
    Args:
        current_state: 'trans' or 'cis'
        wavelength: key from PHOTOSWITCH_DATA
    
    Returns:
        new_state: 'trans' or 'cis'
    """
    if wavelength not in PHOTOSWITCH_DATA:
        return current_state
    
    switch = PHOTOSWITCH_DATA[wavelength]
    
    # Check if this wavelength can drive the switch
    if switch["direction"] == "trans_to_cis" and current_state == "trans":
        if random.random() < switch["quantum_yield"]:
            return "cis"
    elif switch["direction"] == "cis_to_trans" and current_state == "cis":
        if random.random() < switch["quantum_yield"]:
            return "trans"
    
    return current_state

def get_switching_efficiency(initial_state, final_state, wavelength):
    """Calculate the effective switching efficiency."""
    if initial_state == final_state:
        return 0.0
    return PHOTOSWITCH_DATA[wavelength]["quantum_yield"] * 100
