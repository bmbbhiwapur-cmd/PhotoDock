# modules/receptor_library.py

RECEPTORS = {
    "DNA Gyrase (S. aureus)": {
        "pdb_id": "2XCS",
        "description": "Essential enzyme for DNA supercoiling during replication. Target of fluoroquinolone antibiotics.",
        "organism": "Staphylococcus aureus",
        "key_residues": ["SER84", "GLU88", "ARG122", "ASP83"],
        "binding_site": "Quinolone-binding pocket at DNA interface",
    },
    "FtsZ (S. aureus)": {
        "pdb_id": "4DXD",
        "description": "Tubulin homolog essential for bacterial cell division. Forms the Z-ring at the septum.",
        "organism": "Staphylococcus aureus",
        "key_residues": ["GLY196", "ASN263", "THR309", "ARG143"],
        "binding_site": "T7 loop and nucleotide-binding domain",
    },
    "DHFR (E. coli)": {
        "pdb_id": "3G7E",
        "description": "Dihydrofolate reductase. Key enzyme in folate biosynthesis pathway. Target of trimethoprim.",
        "organism": "Escherichia coli",
        "key_residues": ["ASP27", "PHE31", "ILE94", "THR113"],
        "binding_site": "Folate binding pocket with catalytic Asp27",
    },
    "PBP2a (MRSA)": {
        "pdb_id": "6V5S",
        "description": "Penicillin-binding protein 2a from MRSA. Confers beta-lactam antibiotic resistance.",
        "organism": "Methicillin-resistant S. aureus",
        "key_residues": ["SER403", "LYS406", "TYR446", "GLU447"],
        "binding_site": "Transpeptidase active site with catalytic Ser403",
    },
}
