from pathlib import Path
from fastapi import FastAPI, File, UploadFile # pip install fastapi uvicorn python-multipart
from fastapi.responses import FileResponse, JSONResponse
import shutil

app = FastAPI()

# Define the upload directory
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"message": "Image uploaded successfully", "file_name": file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download-image/{file_name}")
async def download_image(file_name: str):
    file_path = UPLOAD_DIR / file_name
    if file_path.exists():
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    else:
        return JSONResponse(status_code=404, content={"error": "File not found"})

# function for cropping each detection, returning the largest one and resizing to a specified size in px
def yolo_and_crop(image_array, torch_model):
    yolo_res = torch_model(image_array)
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
