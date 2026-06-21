# modules/utils.py

import os

def ensure_temp_dir():
    """Ensure the temp directory exists."""
    os.makedirs("temp", exist_ok=True)

def clean_temp():
    """Clean temporary files."""
    import shutil
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))

def calculate_selectivity(dark_energy, light_energy):
    """
    Calculate predicted MIC fold change from binding energy difference.
    ΔG = -RT ln(Kd) → approximate to MIC fold change.
    """
    delta = dark_energy - light_energy
    # Approximate: each 1.36 kcal/mol ≈ 10x potency change at 298K
    fold_change = 10 ** (delta / 1.36)
    return round(fold_change, 0)
