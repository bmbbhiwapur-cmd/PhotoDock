# 💡 PhotoDock IKS

### From Ancient Sun Therapy to Red-Light Precision Antibiotics

---

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**A Computational Platform Integrating Indian Knowledge System with Modern Photopharmacology**

[Live Demo](https://yourusername-photodock-iks.streamlit.app) · [Report Bug](https://github.com/yourusername/PhotoDock-IKS/issues) · [Request Feature](https://github.com/yourusername/PhotoDock-IKS/issues)

</div>

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [The Indian Knowledge System Connection](#the-indian-knowledge-system-connection)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Scientific Background](#scientific-background)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About the Project

**PhotoDock IKS** is an interactive computational platform that bridges the ancient Indian Knowledge System (IKS) of **Sūrya-saṃyoga-cikitsā** (sun-combined therapy) with modern molecular docking and photopharmacology. 

The tool allows researchers and students to:
- Select natural photosensitizers from Indian medicinal plants documented in classical Āyurvedic texts
- Perform molecular docking against bacterial protein targets
- Simulate light-induced photoisomerization of the phytochemicals
- Compare binding affinities between dark (inactive) and light (active) states
- Predict antimicrobial selectivity factors

### Why This Matters

Antimicrobial resistance (AMR) is projected to cause 10 million deaths annually by 2050. The Indian Knowledge System offers a treasure trove of time-tested natural photosensitizers. By computationally redesigning these molecules with modern azobenzene photoswitches, we can create **photoantibiotics** — drugs that remain inactive in the body until activated by harmless red light at the infection site. This spares the gut microbiome, prevents resistance development, and allows the reuse of ancient antimicrobial wisdom.

---

## The Indian Knowledge System Connection

### Sūrya-saṃyoga-cikitsā (सूर्य-संयोग-चिकित्सा)

The classical Āyurvedic texts, particularly the **Caraka Saṃhitā** and **Suśruta Saṃhitā**, describe the therapeutic use of plant-based preparations combined with sunlight (*Ātapa*) for treating skin disorders:

| Plant (Sanskrit Name) | Phytochemical | Traditional Use |
|:----------------------|:--------------|:----------------|
| **Bākuchī** (*Psoralea corylifolia*) | Psoralen | Vitiligo (Śvitra) treated with seed paste + morning sun |
| **Bilva** (*Aegle marmelos*) | Marmelosin | Leukoderma and psoriasis with leaf juice + sunlight |
| **Chitraka** (*Plumbago zeylanica*) | Plumbagin | Skin abscesses with root paste + sun exposure |
| **Mendhī** (*Lawsonia inermis*) | Lawsone | Antifungal skin applications sun-dried for potency |
| **Bijapūra** (*Citrus medica*) | Bergapten | Skin pigmentation restoration with peel oil + sun |

This platform computationally recreates and enhances this ancient principle with molecular precision.

---

## Features

### 🔬 Phase 1: Ligand & Receptor Selection
- **5 curated IKS natural photosensitizers** with detailed metadata
- **4 bacterial protein targets** (DNA Gyrase, FtsZ, DHFR, PBP2a)
- SMILES notation display with molecular properties
- Traditional use documentation for each phytochemical

### 🌑 Phase 2: Dark-State Docking
- Simulated AutoDock Vina molecular docking
- Binding energy calculation in kcal/mol
- Key amino acid residue interaction analysis
- Multiple docking pose display
- 3D molecular visualization placeholder

### 💡 Phase 3: Photo-Isomerization
- **5 wavelength options** (UV 365nm to NIR 750nm)
- Quantum yield-based switching simulation
- Trans ↔ Cis molecular animation
- Photon energy and direction display
- Educational tooltips explaining photochemistry

### ☀️ Phase 4: Light-State Docking & Results
- Side-by-side binding energy comparison
- ΔΔG calculation with selectivity prediction
- MIC fold change estimation
- IKS therapeutic prediction summary
- Residue interaction comparison (bound vs lost)

### 📊 Additional Features
- Real-time progress indicators
- Responsive card-based UI design
- Session state management
- One-click reset for new sessions
- Mobile-responsive layout

---

## Technology Stack

| Technology | Purpose |
|:-----------|:--------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web application framework |
| **RDKit** | Cheminformatics library for molecular manipulation |
| **AutoDock Vina** | Molecular docking engine (simulated in demo) |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Biopython** | PDB file parsing and biological computations |
| **Py3Dmol** | 3D molecular visualization |
| **GitHub** | Version control and collaboration |
| **Streamlit Cloud** | Free deployment platform |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip
- Git

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PhotoDock-IKS.git
cd PhotoDock-IKS
