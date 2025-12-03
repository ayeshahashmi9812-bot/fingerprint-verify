from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fingerprint_processor import fingerprint_verifier
import uvicorn

app = FastAPI(title="Fingerprint Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print(" Loading fingerprint dataset...")
    success = fingerprint_verifier.load_dataset("dataset")
    if success:
        print(f" Dataset loaded! Registered: {fingerprint_verifier.get_registered_count()} persons")
    else:
        print(" Dataset loading failed!")

@app.get("/")
async def root():
    return {
        "message": "Fingerprint Verification API",
        "status": "active", 
        "registered_fingerprints": fingerprint_verifier.get_registered_count()
    }

@app.post("/fingerprint/register")
async def register_fingerprint(
    person_id: str = Form(..., description="Person ID (e.g., person_1)"),
    fingerprint_image: UploadFile = File(..., description="Fingerprint image")
):
    try:
        print(f"Registering fingerprint for: {person_id}")
        
        if not fingerprint_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Please upload an image file")
    
        image_bytes = await fingerprint_image.read()
        
        # Register fingerprint
        success = fingerprint_verifier.register_fingerprint(person_id, image_bytes)
        
        if success:
            return {
                "success": True,
                "message": f"Fingerprint registered for {person_id}",
                "person_id": person_id,
                "registered_count": fingerprint_verifier.get_registered_count()
            }
        else:
            raise HTTPException(status_code=400, detail="Fingerprint registration failed")
            
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fingerprint/verify")
async def verify_fingerprint(
    fingerprint_image: UploadFile = File(..., description="Fingerprint image to verify")
):
    try:
        print("Verifying fingerprint...")
    
        if not fingerprint_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Please upload an image file")
            
        image_bytes = await fingerprint_image.read()
        
        result = fingerprint_verifier.verify_fingerprint(image_bytes)
        
        result["matched"] = bool(result["matched"])
        result["confidence"] = float(result["confidence"])
        print(f" Verification result: {result}")
        return result
        
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return {"matched": False, "confidence": 0.0}

@app.get("/fingerprint/registered")
async def get_registered_fingerprints():
    """Get all registered person IDs"""
    return {
        "registered_count": fingerprint_verifier.get_registered_count(),
        "person_ids": list(fingerprint_verifier.registered_embeddings.keys())
    }

if __name__ == "__main__":
    print("Fingerprint Verification Server Starting...")

    uvicorn.run(app, host="0.0.0.0", port=8000)
