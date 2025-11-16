# preprocessing.py
import os, sys, random, pathlib, io
from glob import glob
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# ================== CONFIG ==================
TARGET_H, TARGET_W = 1024, 512
SEED = 42
# ============================================

try:
    import pydicom
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

random.seed(SEED); np.random.seed(SEED)

def _dicom_first(v):
    try: return v[0]
    except Exception: return v

# --- Core Image Processing Functions ---

def clahe_enhance(img):
    """Applies CLAHE contrast enhancement."""
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def resize_keep_ar_pad(img, target_h, target_w):
    """Resize with aspect ratio kept and zero-padding."""
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    
    # Return canvas for backend, but meta for batch processing
    meta = dict(
        orig_h=h, orig_w=w, target_h=target_h, target_w=target_w,
        scale=scale, top=top, left=left
    )
    return canvas, meta

# ============================================================
# >>> FUNCTION FOR YOUR BACKEND <<<
# ============================================================
def preprocess_single_image_bytes(
    img_bytes: bytes, 
    apply_clahe: bool = True
) -> np.ndarray:
    """
    Applies the "golden" preprocessing pipeline to raw image bytes.
    
    Args:
        img_bytes: Raw bytes from file upload.
        apply_clahe: If True, applies the medianBlur + CLAHE step.
    """
    
    img = None # This will hold our uint8 grayscale image

    # --- 1. Try to read as DICOM ---
    if HAVE_PYDICOM:
        try:
            ds = pydicom.dcmread(io.BytesIO(img_bytes))
            meta = {}
            meta["photometric"] = str(getattr(ds, "PhotometricInterpretation", None))
            meta["window_center"]   = float(_dicom_first(getattr(ds, "WindowCenter", np.nan))) if hasattr(ds, "WindowCenter") else None
            meta["window_width"]    = float(_dicom_first(getattr(ds, "WindowWidth",  np.nan))) if hasattr(ds, "WindowWidth")  else None
            meta["rescale_intercept"] = float(getattr(ds, "RescaleIntercept", 0.0))
            meta["rescale_slope"]     = float(getattr(ds, "RescaleSlope",     1.0))

            arr = ds.pixel_array.astype(np.float32)
            arr = arr * meta["rescale_slope"] + meta["rescale_intercept"]

            if meta["window_center"] is not None and meta["window_width"] is not None:
                wc, ww = meta["window_center"], meta["window_width"]
                lo, hi = wc - ww/2.0, wc + ww/2.0
                arr = np.clip(arr, lo, hi)

            arr = arr - np.min(arr)
            if np.max(arr) > 0: arr = arr / np.max(arr)
            if str(meta["photometric"]).upper() == "MONOCHROME1": arr = 1.0 - arr
            img = (arr * 255.0).clip(0, 255).astype(np.uint8)
        except Exception:
            img = None

    # --- 2. Fallback to standard image (JPG, PNG, etc.) ---
    if img is None:
        try:
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError("cv2.imdecode failed")
        except Exception as e:
            raise ValueError(f"Could not decode image. Not a valid DICOM or image file. Error: {e}")

    # --- 3. Apply CLAHE (or not) ---
    if apply_clahe:
        img_processed = clahe_enhance(img)
    else:
        img_processed = img # Skip CLAHE

    # --- 4. Resize + Pad ---
    # We only need the canvas, not the 'meta' dict
    canvas, _ = resize_keep_ar_pad(img_processed, TARGET_H, TARGET_W)
    return canvas

# ============================================================
# (The rest is your original script for batch-processing)
# ============================================================
def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def collect_images(folder):
    exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.dcm"]
    paths = []
    for e in exts:
        paths += glob(os.path.join(folder, e))
        paths += glob(os.path.join(folder, "**", e), recursive=True)
    return sorted(list({os.path.abspath(p) for p in paths}))

def read_image_with_meta(path):
    """Reads DICOM or normal image and returns uint8 + metadata dict."""
    meta = {
        "is_dicom": False, "photometric": None, "pixel_spacing_row": None,
        "pixel_spacing_col": None, "window_center": None, "window_width": None,
        "rescale_intercept": None, "rescale_slope": None,
    }
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".dcm":
        if not HAVE_PYDICOM: raise RuntimeError("Found .dcm but pydicom not installed.")
        ds = pydicom.dcmread(path)
        meta["is_dicom"] = True
        meta["photometric"] = str(getattr(ds, "PhotometricInterpretation", None))
        if hasattr(ds, "PixelSpacing"):
            try:
                ps = [float(x) for x in _dicom_first(ds.PixelSpacing)]
                meta["pixel_spacing_row"] = ps[0]
                meta["pixel_spacing_col"] = ps[1] if len(ps) > 1 else None
            except Exception: pass
        meta["window_center"]   = float(_dicom_first(getattr(ds, "WindowCenter", np.nan))) if hasattr(ds, "WindowCenter") else None
        meta["window_width"]    = float(_dicom_first(getattr(ds, "WindowWidth",  np.nan))) if hasattr(ds, "WindowWidth")  else None
        meta["rescale_intercept"] = float(getattr(ds, "RescaleIntercept", 0.0))
        meta["rescale_slope"]     = float(getattr(ds, "RescaleSlope",     1.0))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * meta["rescale_slope"] + meta["rescale_intercept"]
        if meta["window_center"] is not None and meta["window_width"] is not None:
            wc, ww = meta["window_center"], meta["window_width"]
            lo, hi = wc - ww/2.0, wc + ww/2.0
            arr = np.clip(arr, lo, hi)
        arr = arr - np.min(arr)
        if np.max(arr) > 0: arr = arr / np.max(arr)
        if str(meta["photometric"]).upper() == "MONOCHROME1": arr = 1.0 - arr
        img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise RuntimeError(f"Cannot read image: {path}")
    return img, meta

def process_images(in_dir, out_dir):
    INPUT_DIR = "/content/roboflow" # Or wherever your batch images are
    OUT_DIR = "/content/roboflow"
    if not in_dir or not os.path.isdir(in_dir):
        print(f"[ERROR] Folder missing: {in_dir}")
        return
    imgs = collect_images(in_dir)
    if not imgs:
        print(f"[INFO] No images found in {in_dir}")
        return
    save_dir = os.path.join(out_dir, "images")
    ensure_dir(save_dir)
    print(f"Found {len(imgs)} images in {in_dir}")
    manifest_rows = []
    for src in tqdm(imgs, desc="Preprocessing"):
        try:
            img, dcm_meta = read_image_with_meta(src)
            img_clahe = clahe_enhance(img)
            canvas, geom = resize_keep_ar_pad(img_clahe, TARGET_H, TARGET_W) 
            
            # --- THIS WAS THE MISSING LINE ---
            base = pathlib.Path(src).stem
            # -----------------------------------
            
            out_name = f"{base}.png"
            cv2.imwrite(os.path.join(save_dir, out_name), canvas)
            
            manifest_rows.append([
                out_name, src,
                geom["orig_h"], geom["orig_w"], geom["target_h"], geom["target_w"],
                geom["scale"], geom["top"], geom["left"],
                dcm_meta["is_dicom"], dcm_meta["photometric"],
                dcm_meta["pixel_spacing_row"], dcm_meta["pixel_spacing_col"],
                dcm_meta["window_center"], dcm_meta["window_width"],
                dcm_meta["rescale_intercept"], dcm_meta["rescale_slope"]
            ])
        except Exception as e:
            print(f"Failed to process {src}: {e}")
            
    if manifest_rows:
        cols = [
            "filename","source_path", "orig_h","orig_w","target_h","target_w",
            "scale","top","left", "is_dicom","photometric",
            "pixel_spacing_row_mm","pixel_spacing_col_mm",
            "window_center","window_width", "rescale_intercept","rescale_slope"
        ]
        df = pd.DataFrame(manifest_rows, columns=cols)
        df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
        print(f"\n✅ Done. Processed images saved to: {save_dir}")
        print(f"📄 Manifest saved at: {os.path.join(out_dir, 'manifest.csv')}")
    else:
        print("⚠️ No images processed.")

def main():
    # Example paths, change these as needed
    INPUT_DIR = "/content/roboflow" 
    OUT_DIR = "/content/roboflow"
    if "<PATH_" in INPUT_DIR or "<PATH_" in OUT_DIR:
        print("Edit INPUT_DIR and OUT_DIR paths at the top of the script.")
        sys.exit(1)
    ensure_dir(OUT_DIR)
    process_images(INPUT_DIR, OUT_DIR)

if __name__ == "__main__":
    main()