from pathlib import Path
from ultralytics import YOLO # pip install ultralyitcs
from fastapi import FastAPI, File, UploadFile # pip install fastapi uvicorn python-multipart
from fastapi.responses import FileResponse, JSONResponse
import shutil, cv2
import numpy as np

app = FastAPI()

UPLOAD_DIR = Path("uploads/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path("uploads/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

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
        num_objects = process_yolo_results(image, results)

        processed_file_path = PROCESSED_DIR / f"processed_{file.filename}"
        
        # Save the processed image (dummy example using original image)
        cv2.imwrite(str(processed_file_path), image)

        return {
            "message": "Image uploaded and processed successfully",
            "file_name": file.filename,
            "image_shape": image.shape,
            "number_of_detected_objects": num_objects,
            "processed_file": str(processed_file_path)
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


def process_yolo_results(image, results):
    """
    Process YOLO results: count objects, crop detected objects,
    and save them to the processed folder.
    """
    num_objects = 0
    for i, result in enumerate(results):
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            # Increment object count
            num_objects += 1
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cropped_object = image[y1:y2, x1:x2]

            # Create a folder for each class
            class_name = f"class_{int(cls)}"
            class_dir = PROCESSED_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save the cropped image
            cropped_file_path = class_dir / f"object_{num_objects}.png"
            cv2.imwrite(str(cropped_file_path), cropped_object)
    
    return num_objects

# function for cropping each detection, returning the largest one and resizing to a specified size in px
def yolo_and_crop(image_array, yolo_model):
    yolo_res = yolo_model(image_array)
    yolo_pd = yolo_res.pandas().xyxy[0]

    num_objects = yolo_pd.shape[0]
    boxes = yolo_pd.iloc[:, 0:4]
    # scores = yolo_pd.iloc[:, 4]  # unused results
    # class_names = yolo_pd.iloc[:, 6]
    # create dictionary to hold count of objects for image name
    img_list = []
    dims_mult = []
    for i in range(num_objects):
        # get box coordinates
        xmin, ymin, xmax, ymax = boxes.iloc[i]
        # crop detection from image (take an additional 5 pixels around all edges)
        cropped_img = image_array[int(ymin) - 0:int(ymax) + 0, int(xmin) - 0:int(xmax) + 0]  # 0 px added to the frame
        mult = cropped_img.shape[0] * cropped_img.shape[1]
        img_list.append(cropped_img)
        dims_mult.append(mult)

    largest_img_indx = dims_mult.index(max(dims_mult))
    largest_img = img_list[largest_img_indx]
    return result


# Run the app with: uvicorn app:app --reload
