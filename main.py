from pathlib import Path
from typing import List, Tuple
from skimage.metrics import structural_similarity as compare_ssim
from ultralytics import YOLO # pip install ultralyitcs
from fastapi import FastAPI, File, UploadFile # pip install fastapi uvicorn python-multipart
from fastapi.responses import FileResponse, JSONResponse
import shutil, cv2
import numpy as np

app = FastAPI()

UPLOAD_DIR = Path("uploads/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DETECTED_DIR = Path("uploads/detected")
DETECTED_DIR.mkdir(exist_ok=True)

# Load a pretrained YOLO model
model_yolo = YOLO("yolo11l.pt")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the file for OpenCV processing
        with file_path.open("rb") as image_file:
            nparr = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        # Detect w/ YOLO
        results = model_yolo(image, stream=True)

        # Process results: count and crop objects
        num_objects, object_classes = process_yolo_results(image, file.filename, results)

        return {
            "message": "Image uploaded and detected successfully",
            "file_name": file.filename,
            "image_shape": image.shape,
            "number_of_detected_objects": num_objects,
            "detected_objects": object_classes  # Include the updated dictionary
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await file.close()


@app.get("/download-image/{file_name}")
async def download_image(file_name: str):
    try:
        file_path = find_image_path(file_name)
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

@app.get("/get-similar-image/{file_name}")
async def get_similar_image(file_name: str):
    try:
        # Find the query file path
        query_file_path = find_image_path(file_name)

        # Load the query image for processing
        query_image = cv2.imread(str(query_file_path))
        if query_image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid query image file"})

        # Search for matches in all uploads (raw and detected)
        matches = compare_image(query_image, UPLOAD_DIR) + compare_image(query_image, DETECTED_DIR)

        # Sort matches by similarity (highest SSIM score first) and limit to top 10
        matches = sorted(matches, key=lambda x: x[1], reverse=True)[:10]

        return {
            "message": "Search completed",
            "query_file_name": file_name,
            "top_matches": [{"file_name": match[0], "similarity_score": match[1]} for match in matches]
        }
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def process_yolo_results(image, file_name, results):
    """
    Process YOLO results: count objects, crop detected objects,
    and save them to the 'detected' folder with their class names.
    """
    class_names = model_yolo.names
    num_objects = 0
    object_classes = {}  # Dictionary to hold object numbers, class names, and saved image names

    for result in results:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            # Increment object count
            num_objects += 1
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cropped_object = image[y1:y2, x1:x2]

            # Get class name
            class_name = class_names[int(cls)]

            # Create a folder for each class
            class_dir = DETECTED_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save the cropped image
            base_name = file_name.rsplit('.', 1)[0]  # Remove file extension
            cropped_file_name = f"{base_name}_object_{num_objects}.png"
            cropped_file_path = class_dir / cropped_file_name
            cv2.imwrite(str(cropped_file_path), cropped_object)

            # Add to dictionary with the saved image name
            object_classes[num_objects] = {
                "class_name": class_name,
                "saved_image_name": cropped_file_name
            }
    
    return num_objects, object_classes

def compare_image(query_image: np.ndarray, search_dir: Path) -> List[Tuple[str, float]]:
    """
    Searches for similar images in the specified directory (including subdirectories).
    Converts images to grayscale and compares using SSIM.
    Returns a list of tuples containing file names and similarity scores.
    """
    matches = []
    # Convert the query image to grayscale
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    for file_path in search_dir.rglob("*.*"):  # Search all files in subdirectories
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:  # Supported image formats
            target_image = cv2.imread(str(file_path))
            if target_image is None:
                continue

            # Convert the target image to grayscale
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

            # Resize images to the same size for comparison
            query_resized = cv2.resize(query_gray, (100, 100))
            target_resized = cv2.resize(target_gray, (100, 100))

            # Calculate similarity using SSIM
            similarity_score, _ = compare_ssim(query_resized, target_resized, full=True)

            # Append the file name and similarity score
            matches.append((str(file_path), similarity_score))

    return matches

def find_image_path(file_name: str) -> Path:
    """
    Finds the image path by searching in raw and detected directories.
    Returns the path if found, otherwise raises a FileNotFoundError.
    """
    import re

    # Check if the file name contains "object_d+"
    if re.search(r"object_\d+", file_name):
        # Search in detected folder
        for class_dir in DETECTED_DIR.iterdir():
            if class_dir.is_dir():
                detected_file_path = class_dir / file_name
                if detected_file_path.exists():
                    return detected_file_path
        raise FileNotFoundError(f"File '{file_name}' not found in detected images.")
    else:
        # Search in raw folder
        raw_file_path = UPLOAD_DIR / file_name
        if raw_file_path.exists():
            return raw_file_path
        raise FileNotFoundError(f"File '{file_name}' not found in raw images.")
