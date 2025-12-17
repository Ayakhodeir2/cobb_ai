# backend_clinical.py - Simplified Clinical Backend
"""
Streamlined backend for clinical Cobb angle measurement.
Fast, reliable predictions without agentic AI overhead.
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple

import torch
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models_unet_resnet import UNet, ResNet50Cobb
from preprocessing import preprocess_single_image_bytes

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNET_WEIGHTS   = "weights/unet_scoliosis8_best.pt"
RESNET_WEIGHTS = "weights/resnet50_unetmask_best.pt"

MASK_THRESH = 0.5
UNET_IMG_SIZE = 512

unet_tf = transforms.Compose([
    transforms.Resize((UNET_IMG_SIZE, UNET_IMG_SIZE)),
    transforms.ToTensor(),
])

resnet_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================================================
# MODEL LOADING
# ============================================================
def load_unet() -> UNet:
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    state = torch.load(UNET_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_resnet() -> ResNet50Cobb:
    model = ResNet50Cobb()
    state = torch.load(RESNET_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

print("Loading models...")
UNET_MODEL = load_unet()
RESNET_MODEL = load_resnet()
print("Models loaded successfully!")

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def get_spine_mask(unet: UNet, pil_img: Image.Image, thr: float = MASK_THRESH) -> np.ndarray:
    """Segment spine from X-ray using U-Net."""
    with torch.no_grad():
        x = unet_tf(pil_img).unsqueeze(0).to(DEVICE)
        logits = unet(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    mask_small = (prob > thr).astype(np.uint8)
    mask = cv2.resize(mask_small, pil_img.size, interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.float32)


def preprocess_for_resnet(img_clahe_np: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Apply mask and prepare image for ResNet."""
    img_np = img_clahe_np.astype(np.float32)
    masked = img_np * mask
    
    if masked.max() > 0:
        masked = masked / 255.0
    
    masked_3ch = np.stack([masked, masked, masked], axis=-1)
    masked_pil = Image.fromarray((masked_3ch * 255).astype(np.uint8))
    
    return resnet_tf(masked_pil)


def predict_cobb_angles(unet: UNet, resnet: ResNet50Cobb, img_bytes: bytes) -> Tuple[float, float]:
    """
    Predict Cobb angles from X-ray image.
    
    Returns:
        (thoracic_angle, lumbar_angle) in degrees
    """
    # Preprocess image
    img_clahe_np = preprocess_single_image_bytes(img_bytes, apply_clahe=True)
    pil_img = Image.fromarray(img_clahe_np)
    
    # Segment spine
    mask = get_spine_mask(unet, pil_img, thr=MASK_THRESH)
    
    # Prepare for angle prediction
    x = preprocess_for_resnet(img_clahe_np, mask)
    x = x.unsqueeze(0).to(DEVICE)
    
    # Predict angles
    with torch.no_grad():
        out = resnet(x)
    
    out_np = out.cpu().numpy()[0]
    thoracic = float(out_np[0])
    lumbar = float(out_np[1])
    
    # Ensure no negative angles (anatomically impossible)
    thoracic = max(0.0, thoracic)
    lumbar = max(0.0, lumbar)
    
    return thoracic, lumbar


# ============================================================
# API SCHEMAS
# ============================================================

class CobbResult(BaseModel):
    filename: str
    thoracic_cobb_deg: float
    lumbar_cobb_deg: float

class CobbResponse(BaseModel):
    results: List[CobbResult]

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Clinical Cobb Angle Measurement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Clinical Cobb Angle Measurement System",
        "version": "1.0",
        "status": "operational",
        "endpoints": {
            "/predict_cobb": "Predict Cobb angles from X-ray images",
            "/health": "Check system health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "models_loaded": True,
        "device": str(DEVICE)
    }

@app.post("/predict_cobb", response_model=CobbResponse)
async def predict_cobb_endpoint(files: List[UploadFile] = File(...)):
    """
    Predict Cobb angles from one or more X-ray images.
    
    Accepts: Multiple image files (JPG, PNG, DICOM)
    Returns: Thoracic and lumbar Cobb angle measurements
    """
    results = []
    
    for f in files:
        content = await f.read()
        try:
            thoracic, lumbar = predict_cobb_angles(UNET_MODEL, RESNET_MODEL, content)
            
            results.append(CobbResult(
                filename=f.filename,
                thoracic_cobb_deg=thoracic,
                lumbar_cobb_deg=lumbar
            ))
            
            print(f"✓ Processed {f.filename}: Thoracic={thoracic:.1f}°, Lumbar={lumbar:.1f}°")
            
        except Exception as e:
            print(f"✗ Error processing {f.filename}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return -1 for failed predictions
            results.append(CobbResult(
                filename=f.filename,
                thoracic_cobb_deg=-1,
                lumbar_cobb_deg=-1
            ))
    
    return CobbResponse(results=results)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  Clinical Cobb Angle Measurement System")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Starting server on http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)