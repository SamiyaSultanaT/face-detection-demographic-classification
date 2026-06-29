import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from backend.app.core.logging import logger

AGE_GROUPS = ["0-2", "4-6", "8-12", "15-20", "21-30", "31-40", "41-50", "51+"]

def age_to_group_idx(age: int) -> int:
    """
    Maps an integer age to one of the 8 age group indices:
    0-2, 4-6, 8-12, 15-20, 21-30, 31-40, 41-50, 51+
    """
    if age <= 3:
        return 0  # "0-2" (includes 3 for continuity)
    elif age <= 7:
        return 1  # "4-6" (includes 7 for continuity)
    elif age <= 13:
        return 2  # "8-12" (includes 13 for continuity)
    elif age <= 20:
        return 3  # "15-20"
    elif age <= 30:
        return 4  # "21-30"
    elif age <= 40:
        return 5  # "31-40"
    elif age <= 50:
        return 6  # "41-50"
    else:
        return 7  # "51+"

def parse_filename(filepath: str):
    """
    Parses UTKFace filenames to extract age and gender.
    Format: [age]_[gender]_[race]_[timestamp].jpg
    Returns (age_group_idx, gender_label) where gender_label is 0 (Male) or 1 (Female)
    """
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if len(parts) < 3:
        return None, None
    try:
        age = int(parts[0])
        gender = int(parts[1])  # 0: Male, 1: Female in UTKFace
        if gender not in [0, 1]:
            return None, None
        
        age_idx = age_to_group_idx(age)
        return age_idx, gender
    except ValueError:
        return None, None

def load_dataset_paths(dataset_dir: str):
    """
    Scans the dataset directory and collects valid image filepaths and labels.
    """
    filepaths = glob.glob(os.path.join(dataset_dir, "*.jpg"))
    if not filepaths:
        # Check subdirectories too (sometimes extracted inside a nested folder)
        filepaths = glob.glob(os.path.join(dataset_dir, "*", "*.jpg"))
        
    valid_paths = []
    age_labels = []
    gender_labels = []
    
    for path in filepaths:
        age_idx, gender = parse_filename(path)
        if age_idx is not None and gender is not None:
            valid_paths.append(path)
            age_labels.append(age_idx)
            gender_labels.append(gender)
            
    logger.info(f"Found {len(valid_paths)} valid images out of {len(filepaths)} total files in {dataset_dir}")
    return np.array(valid_paths), np.array(age_labels), np.array(gender_labels)

def preprocess_image_path(filepath: str, label):
    """
    TensorFlow image loading and preprocessing map function.
    """
    # Read file
    img = tf.io.read_file(filepath)
    # Decode jpeg
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize to 96x96
    img = tf.image.resize(img, [96, 96])
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    return img, label

def get_data_pipelines(dataset_dir: str, batch_size: int = 32, task: str = "gender", test_size: float = 0.2):
    """
    Creates tf.data.Dataset pipelines for training and validation.
    task: "gender" (binary classification) or "age" (multiclass classification)
    """
    paths, ages, genders = load_dataset_paths(dataset_dir)
    if len(paths) == 0:
        raise ValueError(f"No valid images found in {dataset_dir}. Check dataset path.")
        
    if task == "gender":
        labels = genders
    else:
        # For age, convert target to one-hot encoding
        labels = tf.keras.utils.to_categorical(ages, num_classes=8)
        
    # Train/Test Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=test_size, random_state=42
    )
    
    # Create Train Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(train_paths))
    train_ds = train_ds.map(preprocess_image_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create Val Dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(preprocess_image_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds
