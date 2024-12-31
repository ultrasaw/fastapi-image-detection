from modules.utils import process_yolo_results, compare_image, find_image_path

from pathlib import Path
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as compare_ssim # pip install scikit-image
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
from ultralytics import YOLO # pip install ultralytics
from fastapi import FastAPI, File, UploadFile  # pip install fastapi uvicorn python-multipart
from fastapi.responses import FileResponse, JSONResponse
import shutil
import re
import cv2 # pip install opencv-python-headless
import numpy as np

app = FastAPI()

UPLOAD_DIR: Path = Path("uploads/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DETECTED_DIR: Path = Path("uploads/detected")
DETECTED_DIR.mkdir(parents=True, exist_ok=True)

# Load a pretrained YOLO model
model_yolo: YOLO = YOLO("yolo11l.pt")


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path: Path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the file for OpenCV processing
        with file_path.open("rb") as image_file:
            nparr: np.ndarray = np.frombuffer(image_file.read(), np.uint8)
            image: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        # Detect w/ YOLO
        results = model_yolo(image, stream=True)

        # Process results: count and crop objects
        num_objects, object_classes = process_yolo_results(image, file.filename, results, model_yolo, DETECTED_DIR)

        return JSONResponse(
            content={
                "message": "Image uploaded and detected successfully",
                "file_name": file.filename,
                "image_shape": image.shape,
                "number_of_detected_objects": num_objects,
                "detected_objects": object_classes,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await file.close()


@app.get("/download-image/{file_name}")
async def download_image(file_name: str):
    try:
        file_path: Path = find_image_path(file_name, UPLOAD_DIR, DETECTED_DIR)
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@app.get("/get-similar-image/{file_name}")
async def get_similar_image(file_name: str):
    try:
        # Find the query file path
        query_file_path: Path = find_image_path(file_name, UPLOAD_DIR, DETECTED_DIR)

        # Load the query image for processing
        query_image: np.ndarray = cv2.imread(str(query_file_path))
        if query_image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid query image file"})

        # Search for matches in all uploads (raw and detected)
        matches: List[Tuple[str, float]] = compare_image(query_image, UPLOAD_DIR) + compare_image(query_image, DETECTED_DIR)

        # Sort matches by similarity (highest SSIM score first) and limit to top 10
        matches = sorted(matches, key=lambda x: x[1], reverse=True)[:10]

        return JSONResponse(
            content={
                "message": "Search completed",
                "query_file_name": file_name,
                "top_matches": [{"file_name": match[0], "similarity_score": match[1]} for match in matches],
            }
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/list-images/")
async def list_images():
    raw_images = [str(file.name) for file in UPLOAD_DIR.glob("*.*") if file.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    detected_images = [
        str(file.name) 
        for class_dir in DETECTED_DIR.iterdir() if class_dir.is_dir()
        for file in class_dir.glob("*.*") if file.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    return JSONResponse(
        content={
            "raw_images": raw_images,
            "detected_images": detected_images,
            "total_images": len(raw_images) + len(detected_images)
        }
    )

@app.get("/list-images/")
async def list_images():
    """
    Returns a list of all image file names in both uploads/raw and uploads/detected directories.
    """
    raw_images = [str(file.name) for file in UPLOAD_DIR.glob("*.*") if file.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    detected_images = [
        str(file.name) 
        for class_dir in DETECTED_DIR.iterdir() if class_dir.is_dir()
        for file in class_dir.glob("*.*") if file.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    return JSONResponse(
        content={
            "raw_images": raw_images,
            "detected_images": detected_images,
            "total_images": len(raw_images) + len(detected_images)
        }
    )
