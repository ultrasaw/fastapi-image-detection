# fastapi-image-detection
A tiny backend for image object detection w/ YOLO. Included REST endpoints:
- POST to upload an image, detect & classify objects present in the image and save all objects as individual *.png* files.
- GET to return names of the images that were uploaded / created via the POST endpoint.
- GET to compute a similarity score between a given image (specified by image name) and all existing images. The score is computed using *scikit's* ```structural_similarity``` function.
- GET to download an image (specified by image name).

## Local development
### venv
Create a python virtual environment:
```bash
python3 -m venv venv_img
```

### pip
Install dependencies:
```bash
pip install -r requirements.txt
```

Create the dependencies file:

```bash
rm requirements.txt
pip freeze > requirements.txt
```

### run
Start the application locally:
```bash
uvicorn main:app --reload
```

Navigate to http://127.0.0.1:8000/docs in your browser to test the application functionality.

### docker
Docker commands to:
- Build the image;
- Run the application;
- Exec into the container for debugging;
- Remove the image (container needs to be deleted first).
```bash
docker build -t fastapi-image-detection:TAG .

docker run -p 8000:8000 fastapi-image-detection:TAG

docker exec -it fastapi-image-detection:TAG bash

docker image rm fastapi-image-detection:TAG -f
```

## Continuous integration
Please see the *.github/workflows/*  sub-directory for the GitHub Actions pipeline files. Currently implemented stages:
- Static code analysis with Pylint;
- Build & push the application image to the GitHub Container Registry;
- CVE scan of the image using Trivy.

Ideas for improvement:
- Kubernetes manifests validation;
- Flux reconciliation status;
- Workload readiness, i.e. Pods are up & running.

## Infrastructure & continuous delivery
Please see the ```README.md``` file in the *infrastructure/*  sub-directory for instructions on how to set up a k3s Kubernetes cluster on AWS to run the application. This file also includes instructions on setting up continuous delivery with Flux.
