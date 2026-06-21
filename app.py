import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# 1. SERVER INITIALIZATION (PHOTODOCK)
# ==========================================
app = FastAPI(
    title="PhotoDock API (सूर्य-संयोग-चिकित्सा)",
    description="From Ancient Sun Therapy to Red-Light Precision Antibiotics. Developed by TLCS.",
    version="1.0.0"
)

# ==========================================
# 2. DATA MODELS (Pydantic)
# ==========================================
class Phytochemical(BaseModel):
    name: str
    smiles: str
    origin: str
    activity: str

class MutationRequest(BaseModel):
    compound_name: str
    native_smiles: str

class MutationResponse(BaseModel):
    status: str
    native_smiles: str
    azo_scaffold_smiles: Optional[str]
    message: str

# ==========================================
# 3. PHASE 1: TRIBAL ZONE KNOWLEDGE DATABASE
# ==========================================
@app.get("/api/v1/library/phytochemicals", response_model=List[Phytochemical])
async def get_phytochemical_library():
    """Returns curated phytochemicals from tribal zones with native C=C bridges."""
    return [
        {"name": "Pterostilbene", "smiles": "COc1cc(C=Cc2ccc(O)cc2)cc(OC)c1", "origin": "Tribal Zone Ethnopharmacology", "activity": "Antidiabetic / Precision Antibiotic"},
        {"name": "Resveratrol", "smiles": "Oc1cc(O)cc(C=Cc2ccc(O)cc2)c1", "origin": "Natural botanical extracts", "activity": "Antidiabetic / Anticancer"}
    ]

# ==========================================
# 4. PHASE 3: SURYA-SANYOG AZOLOGIZATION ENGINE
# ==========================================
class AzologizerEngine:
    def __init__(self):
        # Target the C=C bridge
        self.reaction_smarts = '[c:1]-[CH1:2]=[CH1:3]-[c:4]>>[c:1]-[N:2]=[N:3]-[c:4]'
        self.rxn = AllChem.ReactionFromSmarts(self.reaction_smarts)

    def process(self, native_smiles: str):
        native_mol = Chem.MolFromSmiles(native_smiles)
        if not native_mol: return None
        
        products = self.rxn.RunReactants((native_mol,))
        if not products: return None
        
        azo_mol = products[0][0]
        Chem.SanitizeMol(azo_mol)
        return Chem.MolToSmiles(azo_mol)

engine = AzologizerEngine()

@app.post("/api/v1/mutate", response_model=MutationResponse)
async def mutate_phytochemical(request: MutationRequest):
    """Detects stilbene bridges and computationally mutates them into light-activated azo switches."""
    azo_smiles = engine.process(request.native_smiles)
    
    if azo_smiles:
        return MutationResponse(
            status="success",
            native_smiles=request.native_smiles,
            azo_scaffold_smiles=azo_smiles,
            message=f"Successfully applied Surya-Sanyog mutation to {request.compound_name}. Azo-bridge embedded."
        )
    else:
        raise HTTPException(status_code=400, detail="No viable C=C bridge found for azologization.")

# ==========================================
# 5. SERVER HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {"message": "Welcome to the PhotoDock (सूर्य-संयोग-चिकित्सा) API Engine."}
