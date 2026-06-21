# modules/ligand_library.py

IKS_LIGANDS = {
    "Plumbagin (Chitraka)": {
        "smiles": "CC1=CC(=O)c2c(C1=O)cccc2O",
        "source_plant": "Plumbago zeylanica",
        "iks_use": "Skin infections, abscesses with sun exposure (Surya-samyoga-cikitsa)",
        "target_class": "DNA Gyrase",
        "mechanism": "DNA intercalation inhibitor",
        "molecular_weight": 188.18,
        "logP": 1.2,
    },
    "Marmelosin (Bilva)": {
        "smiles": "CC(=CCOc1c2C=CC(=O)Oc2cc3c1C=CO3)C",
        "source_plant": "Aegle marmelos",
        "iks_use": "Leukoderma (Shvitra), psoriasis with sunlight",
        "target_class": "FtsZ Cell Division Protein",
        "mechanism": "Inhibits FtsZ polymerization",
        "molecular_weight": 270.28,
        "logP": 3.5,
    },
    "Lawsone (Mendhi)": {
        "smiles": "O=C1C=CC(=O)c2ccccc12",
        "source_plant": "Lawsonia inermis",
        "iks_use": "Antifungal skin paste, sun-dried for antimicrobial action",
        "target_class": "Cytochrome bd Oxidase",
        "mechanism": "Electron transport chain disruptor",
        "molecular_weight": 174.15,
        "logP": 1.8,
    },
    "Psoralen (Bakuchi)": {
        "smiles": "O=C1C=CC2=CC=CC3=C2C1=CO3",
        "source_plant": "Psoralea corylifolia",
        "iks_use": "Classical vitiligo (Shvitra) treatment with morning sun",
        "target_class": "DNA (Thymidine cross-linking)",
        "mechanism": "UV-activated DNA photoadduct formation",
        "molecular_weight": 186.16,
        "logP": 1.9,
    },
    "Bergapten (Bijapura)": {
        "smiles": "COc1c2C=CC(=O)Oc2cc3c1C=CO3",
        "source_plant": "Citrus medica",
        "iks_use": "Skin depigmentation restoration with sun exposure",
        "target_class": "DNA (Cross-linking)",
        "mechanism": "5-Methoxypsoralen photoadduct formation",
        "molecular_weight": 216.19,
        "logP": 2.1,
    },
}
