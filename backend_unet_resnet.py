# backend.py - CORRECTLY FIXED VERSION
import math
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models_unet_resnet import UNet, ResNet50Cobb
from preprocessing import preprocess_single_image_bytes, TARGET_H, TARGET_W

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNET_WEIGHTS   = "weights/unet_scoliosis8_best.pt"
RESNET_WEIGHTS = "weights/resnet50_unetmask_best.pt"

MASK_THRESH = 0.5

# U-Net transforms (matches training)
UNET_IMG_SIZE = 512 
unet_tf = transforms.Compose([
    transforms.Resize((UNET_IMG_SIZE, UNET_IMG_SIZE)),
    transforms.ToTensor(),
])

# ============================================================
# CRITICAL FIX: Use EXACT training transforms
# ============================================================
# Training used: Resize((224, 224)) - NOT Resize(256) + CenterCrop(224)!
resnet_tf = transforms.Compose([
    transforms.Resize((224, 224)),  # ← THIS WAS THE BUG
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

UNET_MODEL   = load_unet()
RESNET_MODEL = load_resnet()

# ============================================================
# EXACT REPLICATION OF TRAINING PREPROCESSING
# ============================================================

def get_spine_mask(unet: UNet, pil_img: Image.Image, thr: float = MASK_THRESH) -> np.ndarray:
    """
    Replicates get_spine_mask_for_resnet from training.
    
    Args:
        unet: The loaded U-Net model.
        pil_img: The (1024, 512) CLAHE image as PIL.Image (mode='L').
        thr: Threshold for hard mask.
        
    Returns:
        A (1024, 512) float32 mask with values 0.0 or 1.0
    """
    with torch.no_grad():
        x = unet_tf(pil_img).unsqueeze(0).to(DEVICE)
        logits = unet(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    # Threshold to binary (512x512)
    mask_small = (prob > thr).astype(np.uint8)
    
    # Resize to original size
    mask = cv2.resize(mask_small, pil_img.size, interpolation=cv2.INTER_NEAREST)
    
    # Return as float32 with values 0.0 or 1.0 (matches training)
    return mask.astype(np.float32)


def preprocess_for_resnet(
    img_clahe_np: np.ndarray, 
    mask: np.ndarray,
    debug: bool = False
) -> torch.Tensor:
    """
    EXACT replication of training preprocessing in CobbUNetResNetDataset.__getitem__
    
    Training code:
        img_np = np.array(img, dtype=np.float32)  # 0-255
        masked = img_np * mask                     # 0 outside spine
        if masked.max() > 0:
            masked = masked / 255.0                # normalize to 0-1
        masked_3ch = np.stack([masked, masked, masked], axis=-1)
        masked_pil = Image.fromarray((masked_3ch * 255).astype(np.uint8))
        x = self.transform(masked_pil)
    
    Args:
        img_clahe_np: (1024, 512) uint8 CLAHE image (0-255)
        mask: (1024, 512) float32 mask (0.0 or 1.0)
        debug: If True, saves intermediate images
        
    Returns:
        Preprocessed tensor ready for ResNet
    """
    # Step 1: Convert to float32 (like training)
    img_np = img_clahe_np.astype(np.float32)  # 0-255 range
    
    # Step 2: Apply mask (zeros out non-spine regions)
    masked = img_np * mask  # Still 0-255 range where mask=1
    
    if debug:
        print(f"After masking: min={masked.min():.1f}, max={masked.max():.1f}")
    
    # Step 3: Normalize to 0-1 (like training)
    if masked.max() > 0:
        masked = masked / 255.0
    
    if debug:
        print(f"After normalization: min={masked.min():.3f}, max={masked.max():.3f}")
    
    # Step 4: Stack to 3 channels (0-1 range)
    masked_3ch = np.stack([masked, masked, masked], axis=-1)
    
    # Step 5: Convert back to uint8 for PIL (0-255)
    masked_pil = Image.fromarray((masked_3ch * 255).astype(np.uint8))
    
    if debug:
        masked_pil.save("/tmp/debug_masked_for_resnet.png")
        print(f"Saved masked image to /tmp/debug_masked_for_resnet.png")
    
    # Step 6: Apply ResNet transforms (Resize to 224x224, Normalize)
    tensor = resnet_tf(masked_pil)
    
    return tensor

# ============================================================
# INFERENCE PIPELINE
# ============================================================

def predict_cobb_from_bytes(
    unet: UNet, 
    resnet: ResNet50Cobb, 
    img_bytes: bytes,
    debug: bool = False
):
    """
    Complete inference pipeline: bytes → Cobb angles.
    
    Pipeline:
    1. Preprocess raw bytes to (1024, 512) CLAHE image
    2. Get spine mask from U-Net
    3. Apply mask to CLAHE image
    4. Normalize and transform for ResNet
    5. Predict Cobb angles
    
    Args:
        unet: Loaded U-Net model
        resnet: Loaded ResNet model
        img_bytes: Raw image bytes
        debug: If True, saves intermediate images and prints info
        
    Returns:
        (thoracic_angle, lumbar_angle)
    """
    # Step 1: Preprocess to (1024, 512) CLAHE image (uint8, 0-255)
    img_clahe_np = preprocess_single_image_bytes(
        img_bytes, 
        apply_clahe=True
    )
    
    if debug:
        print(f"\n=== DEBUG INFO ===")
        print(f"CLAHE image: shape={img_clahe_np.shape}, dtype={img_clahe_np.dtype}")
        print(f"CLAHE range: [{img_clahe_np.min()}, {img_clahe_np.max()}]")
        cv2.imwrite("/tmp/debug_01_clahe.png", img_clahe_np)
    
    # Step 2: Convert to PIL for U-Net (mode='L' grayscale)
    pil_img = Image.fromarray(img_clahe_np)
    
    # Step 3: Get spine mask (float32, values 0.0 or 1.0)
    mask = get_spine_mask(unet, pil_img, thr=MASK_THRESH)
    
    if debug:
        print(f"Mask: shape={mask.shape}, dtype={mask.dtype}")
        print(f"Mask range: [{mask.min()}, {mask.max()}]")
        print(f"Mask non-zero: {np.count_nonzero(mask)} / {mask.size} pixels ({100*np.count_nonzero(mask)/mask.size:.1f}%)")
        cv2.imwrite("/tmp/debug_02_mask.png", (mask * 255).astype(np.uint8))
    
    # Step 4: Prepare ResNet input (using EXACT training preprocessing)
    x = preprocess_for_resnet(img_clahe_np, mask, debug=debug)
    x = x.unsqueeze(0).to(DEVICE)
    
    if debug:
        print(f"ResNet input: shape={x.shape}")
        print(f"ResNet input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"ResNet input mean: {x.mean().item():.3f}")
    
    # Step 5: Predict Cobb angles
    with torch.no_grad():
        out = resnet(x)
    
    out_np = out.cpu().numpy()[0]
    thoracic = float(out_np[0])
    lumbar = float(out_np[1])
    
    if debug:
        print(f"\n=== PREDICTIONS ===")
        print(f"Thoracic: {thoracic:.2f}°")
        print(f"Lumbar: {lumbar:.2f}°")
        print(f"===================\n")
    
    return thoracic, lumbar

# ============================================================
# FASTAPI
# ============================================================

class CobbResult(BaseModel):
    filename: str
    thoracic_cobb_deg: float
    lumbar_cobb_deg: float

class CobbResponse(BaseModel):
    results: List[CobbResult]

app = FastAPI(title="Cobb Angle AI – UNet + ResNet50")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Cobb Angle backend is running. Use POST /predict_cobb to make predictions."}

@app.post("/predict_cobb", response_model=CobbResponse)
async def predict_cobb_endpoint(files: List[UploadFile] = File(...)):
    """
    Predict Cobb angles for uploaded spine X-ray images.
    
    Accepts multiple files and returns predictions for each.
    """
    results: List[CobbResult] = []
    
    for i, f in enumerate(files):
        content = await f.read()
        try:
            # Debug first image only
            debug = (i == 0)
            thor, lum = predict_cobb_from_bytes(
                UNET_MODEL, 
                RESNET_MODEL, 
                content,
                debug=debug
            )
        except Exception as e:
            print(f"\n❌ ERROR processing {f.filename}: {e}")
            import traceback
            traceback.print_exc()
            thor, lum = -1.0, -1.0

        results.append(
            CobbResult(
                filename=f.filename,
                thoracic_cobb_deg=thor,
                lumbar_cobb_deg=lum,
            )
        )
    
    return CobbResponse(results=results)


@app.post("/debug_predict")
async def debug_predict_endpoint(file: UploadFile = File(...)):
    """
    Debug endpoint that returns detailed preprocessing information.
    
    Use this to diagnose issues with individual images.
    Saves intermediate images to /tmp/ directory.
    """
    content = await file.read()
    
    try:
        thor, lum = predict_cobb_from_bytes(
            UNET_MODEL, 
            RESNET_MODEL, 
            content,
            debug=True
        )
        
        return {
            "filename": file.filename,
            "thoracic": thor,
            "lumbar": lum,
            "status": "success",
            "message": "Check server console for detailed debug info. Intermediate images saved to /tmp/",
            "debug_images": [
                "/tmp/debug_01_clahe.png",
                "/tmp/debug_02_mask.png", 
                "/tmp/debug_masked_for_resnet.png"
            ]
        }
    except Exception as e:
        import traceback
        return {
            "filename": file.filename,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)