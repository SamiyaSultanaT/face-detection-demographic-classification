import os
import random
import numpy as np
import cv2
from backend.app.core.logging import logger

def generate_mock_dataset(target_dir: str, count: int = 150):
    """
    Generates a synthetic dataset of mock images with filenames formatted like UTKFace.
    This enables the student to run and test the training scripts locally 
    without having to wait for a 100MB+ download.
    """
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Generating {count} mock UTKFace images in '{target_dir}' for testing...")
    
    # We will generate simple square images with shapes/text to act as dummy faces
    for i in range(count):
        # Random age between 1 and 80
        age = random.randint(1, 80)
        # Random gender: 0 (Male) or 1 (Female)
        gender = random.randint(0, 1)
        # Random race: 0 to 4
        race = random.randint(0, 4)
        # Random timestamp
        timestamp = f"20170116171212{i:03d}"
        
        filename = f"{age}_{gender}_{race}_{timestamp}.jpg"
        filepath = os.path.join(target_dir, filename)
        
        # Create a dummy image (RGB random colored box)
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        cv2.rectangle(img, (10, 10), (86, 86), color, -1)
        # Draw a smaller inner box representing a "face"
        cv2.rectangle(img, (30, 30), (66, 66), (color[2], color[0], color[1]), -1)
        # Add some noise to prevent models from learning simple duplicate features
        noise = np.random.randint(0, 20, (96, 96, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(filepath, img)
        
    logger.info("Mock dataset generation complete!")

def show_download_instructions():
    """
    Prints instructions for downloading the full UTKFace dataset.
    """
    instructions = """
======================================================================
DOWNLOAD INSTRUCTIONS FOR THE REAL UTKFace DATASET
======================================================================
To train the models on real human faces, download the UTKFace dataset:

1. Visit Kaggle's UTKFace dataset page:
   https://www.kaggle.com/datasets/jangedajange/utkface-new

2. Download the ZIP file (approx. 110 MB) containing "UTKFace".

3. Extract the contents (a folder containing thousands of .jpg files).

4. Place all .jpg files inside a directory (e.g., 'backend/ml_training/dataset/').

5. Run the training scripts pointing to this directory:
   python train_gender.py --dataset ./dataset
   python train_age.py --dataset ./dataset
======================================================================
"""
    print(instructions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download/Generate UTKFace dataset.")
    parser.add_argument("--mock", action="store_true", help="Generate mock dataset.")
    parser.add_argument("--dir", type=str, default="./dataset_mock", help="Target directory for dataset.")
    args = parser.parse_args()
    
    if args.mock:
        generate_mock_dataset(args.dir)
    else:
        show_download_instructions()
        # By default, generate mock data as well so the pipeline is ready to run
        generate_mock_dataset(args.dir)
