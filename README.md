# fastapi-image-detection
## a tiny backend for image object detection w/ YOLO

## venv
```bash
python3 -m venv venv_img
```

## pip
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

## start the application
```bash
uvicorn main:app --reload
```

## docker
```bash
docker build -t fastapi-image-detection:0.0.1 .
docker run -p 8000:8000 fastapi-image-detection:0.0.1

docker exec -it fastapi-image-detection:0.0.1 bash

docker image rm fastapi-image-detection:0.0.1 -f
```
