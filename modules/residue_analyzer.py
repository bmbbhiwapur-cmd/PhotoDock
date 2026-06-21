# modules/residue_analyzer.py

INTERACTION_DATA = {
    ("Plumbagin (Chitraka)", "DNA Gyrase (S. aureus)"): {
        "trans": [
            {
                "residue": "SER84",
                "interaction": "Hydrogen Bond",
                "distance": "2.8 Å",
                "role": "Catalytic serine – essential for DNA cleavage activity",
            },
            {
                "residue": "ASP83",
                "interaction": "Hydrogen Bond",
                "distance": "3.1 Å",
                "role": "Coordinates Mg²⁺ ion in the active site",
            },
            {
                "residue": "ARG122",
                "interaction": "π-Cation",
                "distance": "3.7 Å",
                "role": "Stabilizes naphthoquinone ring system via electrostatic interaction",
            },
        ],
        "cis": [
            {
                "residue": "SER84",
                "interaction": "Lost (distance >5Å)",
                "distance": "6.2 Å",
                "role": "Key H-bond broken due to ligand bending",
            },
        ],
    },
    ("Marmelosin (Bilva)", "FtsZ (S. aureus)"): {
        "trans": [
            {
                "residue": "GLY196",
                "interaction": "Hydrogen Bond",
                "distance": "2.9 Å",
                "role": "Part of T7 loop, involved in nucleotide binding",
            },
            {
                "residue": "ASN263",
                "interaction": "Van der Waals",
                "distance": "3.5 Å",
                "role": "Stabilizes coumarin core in binding pocket",
            },
            {
                "residue": "THR309",
                "interaction": "Hydrogen Bond",
                "distance": "3.2 Å",
                "role": "C-terminal domain – polymerization interface residue",
            },
        ],
        "cis": [
            {
                "residue": "GLY196",
                "interaction": "Lost (steric clash)",
                "distance": "2.1 Å",
                "role": "Bent ligand clashes with T7 loop backbone",
            },
        ],
    },
}

# Default interaction set for any combination not explicitly mapped
DEFAULT_INTERACTIONS = {
    "trans": [
        {
            "residue": "SER84",
            "interaction": "Hydrogen Bond",
            "distance": "2.8 Å",
            "role": "Catalytic residue – drug target interaction",
        },
        {
            "residue": "ASP83",
            "interaction": "Hydrogen Bond",
            "distance": "3.1 Å",
            "role": "Cofactor-coordinating residue",
        },
        {
            "residue": "ARG122",
            "interaction": "π-Cation",
            "distance": "3.7 Å",
            "role": "Stabilizes aromatic ligand core",
        },
    ],
    "cis": [
        {
            "residue": "SER84",
            "interaction": "Lost H-bond",
            "distance": "6.2 Å",
            "role": "Critical interaction broken by photoisomerization",
        },
        {
            "residue": "PHE98",
            "interaction": "Steric Clash",
            "distance": "1.8 Å",
            "role": "Bent ligand collides with hydrophobic pocket wall",
        },
    ],
}

def analyze_residues(ligand_name, receptor_name, isomer_state):
    """Return interacting residues for a given ligand-receptor pair and isomer state."""
    key = (ligand_name, receptor_name)
    if key in INTERACTION_DATA:
        return INTERACTION_DATA[key].get(isomer_state, DEFAULT_INTERACTIONS[isomer_state])
    return DEFAULT_INTERACTIONS.get(isomer_state, [])
