from pathlib import Path
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
            "detected_objects": object_classes  # Include the dictionary
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await file.close()


@app.get("/download-image/{file_name}")
async def download_image(file_name: str):
    file_path = UPLOAD_DIR / file_name
    if file_path.exists():
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    else:
        return JSONResponse(status_code=404, content={"error": "File not found"})


def process_yolo_results(image, file_name, results):
    """
    Process YOLO results: count objects, crop detected objects,
    and save them to the 'detected' folder with their class names.
    """
    class_names = model_yolo.names
    num_objects = 0
    object_classes = {}  # Dictionary to hold object numbers and class names

    for result in results:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            # Increment object count
            num_objects += 1
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cropped_object = image[y1:y2, x1:x2]

            # Get class name
            class_name = class_names[int(cls)]
            object_classes[num_objects] = class_name  # Add to dictionary

            # Create a folder for each class
            class_dir = DETECTED_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save the cropped image
            base_name = file_name.rsplit('.', 1)[0]  # Remove file extension
            cropped_file_path = class_dir / f"{base_name}_object_{num_objects}.png"
            cv2.imwrite(str(cropped_file_path), cropped_object)
    
    return num_objects, object_classes
