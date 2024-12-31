from pathlib import Path
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as compare_ssim
from ultralytics import YOLO  # pip install ultralyitcs
from fastapi import FastAPI, File, UploadFile  # pip install fastapi uvicorn python-multipart
from fastapi.responses import FileResponse, JSONResponse
import shutil
import re
import cv2
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
        num_objects, object_classes = process_yolo_results(image, file.filename, results)

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
        file_path: Path = find_image_path(file_name)
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@app.get("/get-similar-image/{file_name}")
async def get_similar_image(file_name: str):
    try:
        # Find the query file path
        query_file_path: Path = find_image_path(file_name)

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

def process_yolo_results(image: np.ndarray, file_name: str, results) -> Tuple[int, Dict[int, Dict[str, str]]]:
    """
    Process YOLO results: count objects, crop detected objects,
    and save them to the 'detected' folder with their class names.
    """
    class_names: List[str] = model_yolo.names
    num_objects: int = 0
    object_classes: Dict[int, Dict[str, str]] = {}

    for result in results:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            num_objects += 1
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cropped_object: np.ndarray = image[y1:y2, x1:x2]

            class_name: str = class_names[int(cls)]

            # Create a folder for each class
            class_dir: Path = DETECTED_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save the cropped image
            base_name: str = file_name.rsplit(".", 1)[0]
            cropped_file_name: str = f"{base_name}_object_{num_objects}.png"
            cropped_file_path: Path = class_dir / cropped_file_name
            cv2.imwrite(str(cropped_file_path), cropped_object)

            object_classes[num_objects] = {
                "class_name": class_name,
                "saved_image_name": cropped_file_name,
            }

    return num_objects, object_classes


def compare_image(query_image: np.ndarray, search_dir: Path) -> List[Tuple[str, float]]:
    """
    Searches for similar images in the specified directory (including subdirectories).
    Converts images to grayscale and compares using SSIM.
    """
    matches: List[Tuple[str, float]] = []
    query_gray: np.ndarray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    for file_path in search_dir.rglob("*.*"):
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            target_image: np.ndarray = cv2.imread(str(file_path))
            if target_image is None:
                continue

            target_gray: np.ndarray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

            query_resized: np.ndarray = cv2.resize(query_gray, (100, 100))
            target_resized: np.ndarray = cv2.resize(target_gray, (100, 100))

            similarity_score: float
            similarity_score, _ = compare_ssim(query_resized, target_resized, full=True)

            matches.append((str(file_path), similarity_score))

    return matches


def find_image_path(file_name: str) -> Path:
    """
    Finds the image path by searching in raw and detected directories.
    Returns the path if found, otherwise raises a FileNotFoundError.
    """
    if re.search(r"object_\d+", file_name):
        for class_dir in DETECTED_DIR.iterdir():
            if class_dir.is_dir():
                detected_file_path: Path = class_dir / file_name
                if detected_file_path.exists():
                    return detected_file_path
        raise FileNotFoundError(f"File '{file_name}' not found in detected images.")
    else:
        raw_file_path: Path = UPLOAD_DIR / file_name
        if raw_file_path.exists():
            return raw_file_path
        raise FileNotFoundError(f"File '{file_name}' not found in raw images.")
