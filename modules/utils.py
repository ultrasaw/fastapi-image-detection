from pathlib import Path
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np
import re

def process_yolo_results(image: np.ndarray, file_name: str, results, model, detected_dir: Path) -> Tuple[int, Dict[int, Dict[str, str]]]:
    """
    Process YOLO results: count objects, crop detected objects,
    and save them to the detected folder with their class names.
    """
    class_names: List[str] = model.names
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
            class_dir: Path = detected_dir / class_name
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


def find_image_path(file_name: str, upload_dir: Path, detected_dir: Path) -> Path:
    """
    Finds the image path by searching in raw and detected directories.
    Returns the path if found, otherwise raises a FileNotFoundError.
    """
    if re.search(r"object_\d+", file_name):
        for class_dir in detected_dir.iterdir():
            if class_dir.is_dir():
                detected_file_path: Path = class_dir / file_name
                if detected_file_path.exists():
                    return detected_file_path
        raise FileNotFoundError(f"File '{file_name}' not found in detected images.")
    else:
        raw_file_path: Path = upload_dir / file_name
        if raw_file_path.exists():
            return raw_file_path
        raise FileNotFoundError(f"File '{file_name}' not found in raw images.")
