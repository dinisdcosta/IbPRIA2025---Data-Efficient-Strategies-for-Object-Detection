import os
from random import sample, shuffle
import shutil
from utils.object_detection import train_yolov12, detect_yolov12, test_yolov12, train_yolov5, detect_yolov5, test_yolov5

def get_split(train_size=0.6, val_size=0.2, test_size=0.2, dataset_size=200):
    """
    Splits the dataset indices into train and validation sets, excluding test indices.

    Args:
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        dataset_size (int): Total number of samples in the dataset.

    Returns:
        tuple: (train_indices, val_indices)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
    test_indices = [int(file.split('.')[0]) for file in os.listdir(os.path.join("dataset/test")) if file != "classes.txt"]
    indices = list(range(dataset_size))
    indices = [idx for idx in indices if idx not in test_indices]
    shuffle(indices)
    n_train = int(train_size * dataset_size)
    n_val = int(val_size * dataset_size)
    return indices[:n_train], indices[n_train:]

def clear_run_data():
    """
    Removes all files and subdirectories from the 'dataset/run' directory and recreates
    empty 'train', 'val', and 'test' subdirectories.

    This function is useful for resetting the dataset splits before creating new ones.
    """
    base_path = os.path.join('dataset', 'run')
    shutil.rmtree(base_path, ignore_errors=True)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'test'), exist_ok=True)
    
def split_dataset(dataset="improved", train_split=[], val_split=[], test_split=[]):
    clear_run_data()
    
    def copy_files(file_list, split):
        split_dir = os.path.join('dataset', 'run', split)
        for file in file_list:
            for file_type in ['.jpg', '.txt']:
                src = os.path.join('dataset', dataset if split != 'test' else 'improved', f'{file}{file_type}')
                dst = os.path.join(split_dir, f'{file}{file_type}')
                shutil.copy(src, dst)
    
    copy_files(train_split, 'train')
    copy_files(val_split, 'val')

def get_avg_conf_file(file_path, mode='confidence'):
    """
    Calculates the average confidence score or count of detections from a label file.

    Args:
        file_path (str): Path to the label file.
        mode (str): 'confidence' to return average confidence, 'count' to return number of detections.

    Returns:
        float or int: Average confidence (float) or count (int). Returns 0 if file is empty or error occurs.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            if mode == 'count':
                # Return the number of lines (detections) in the file
                return len(lines)
            # Calculate average confidence from the 6th column (index 5)
            total_conf = sum(float(line.strip().split()[5]) for line in lines)
            return total_conf / len(lines) if lines else 0
    except Exception:
        # Return 0 if file can't be read or is empty
        return 0

def get_new_batch_idx(random=False, al='confidence', project='.temp_batch', name='temp_detect'):
    """
    Returns a new batch of indices for training/validation, either randomly shuffled or sorted by average confidence.

    Args:
        random (bool): If True, returns a randomly shuffled list of available indices.
        al (str): Mode for confidence calculation ('confidence' or 'count').
        project (str): Path to the project directory containing label files.
        name (str): Name of the batch directory inside the project.

    Returns:
        list: List of indices for the new batch, either shuffled or sorted by confidence.
    """
    train_dir = os.path.join('dataset', 'run', 'train')
    val_dir = os.path.join('dataset', 'run', 'val')

    # Get indices (without file extension) for current train and val sets
    train_idx = [file[:-4] for file in os.listdir(train_dir) if file.endswith('.jpg')]
    val_idx = [file[:-4] for file in os.listdir(val_dir) if file.endswith('.jpg')]

    # Get test indices from test set (as integers)
    test_idx = [int(file.split('.')[0]) for file in os.listdir(os.path.join("dataset/test")) if file != "classes.txt"]

    # Get all possible indices from the original dataset (as integers)
    train_val_idx = [int(file.split('.')[0]) for file in os.listdir(os.path.join('dataset', 'original')) if file.endswith('.jpg')]

    # Exclude test indices to get available indices for new batch
    available_idx = [idx for idx in train_val_idx if idx not in test_idx and idx not in train_idx and idx not in val_idx]

    if random:
        # Shuffle and return available indices if random selection is requested
        shuffle(available_idx)
        return available_idx

    # Path to directory containing label files with confidence scores
    confidences_path = os.path.join(project, name, 'labels')

    # Evaluate each available index by its average confidence or count
    evaluations = [
        (score, get_avg_conf_file(os.path.join(confidences_path, f'{score}.txt'), mode=al))
        for score in available_idx
    ]

    # Sort by the evaluation score (ascending order)
    evaluations.sort(key=lambda x: x[1])

    # Return only the indices, sorted by their evaluation score
    return [file[0] for file in evaluations]

def move_img_to_detect():
    """
    Moves images not present in the test, train, or val sets to the 'dataset/detect' directory.

    This function identifies available image indices (not in test/train/val), creates the
    'dataset/detect' directory if it doesn't exist, and moves the corresponding images from
    'dataset/original' to 'dataset/detect'.

    Returns:
        str: Path to the 'dataset/detect' directory containing the moved images.
    """
    available_idx = get_new_batch_idx(random=True)    # Get available indices for detection that are not in train/val/test
    to_detect_dir = os.path.join('dataset', 'detect') # Directory to move images for detection
    shutil.rmtree(to_detect_dir, ignore_errors=True)  # Clear existing detect directory
    os.makedirs(to_detect_dir, exist_ok=True)         # Create detect directory if it doesn't exist

    # Move each available image to the detect directory
    for idx in available_idx:
        src_img = os.path.join('dataset', 'original', f'{idx}.jpg')
        if os.path.exists(src_img):
            shutil.move(src_img, to_detect_dir)

    return to_detect_dir

def new_batch(
    random=False,
    model='v12',
    weights="''",
    al='confidence',
    project='.temp_batch',
    name='temp_detect',
    train_images=10,
    val_images=3,
    dataset='improved'
):
    """
    Creates a new batch of training and validation images, either randomly or using active learning
    (e.g., by confidence or count). Moves images not in train/val/test to a detection directory,
    runs detection if needed, and selects the next batch.

    Args:
        random (bool): If True, selects indices randomly. If False, uses active learning.
        model (str): Model type to use for detection ('v12' or 'v5').
        weights (str): Path to model weights for detection.
        al (str): Active learning mode ('confidence' or 'count').
        project (str): Path to the project directory for detection results.
        name (str): Name of the detection batch directory.
        train_images (int): Number of images to add to the training set.
        val_images (int): Number of images to add to the validation set.
        dataset (str): Name of the dataset directory to use for copying files (e.g., 'improved' or 'original').
    """
    if random:
        best_idx = get_new_batch_idx(random=True) # Randomly select indices for the new batch
    else:
        detection_dir = move_img_to_detect()  # Move available images for detection
        if model == 'v12':
            detect_yolov12(source=detection_dir, project=project, name=name, weights=weights)
        elif model == 'v5':
            detect_yolov5(source=detection_dir, project=project, name=name, weights=weights)
        else:
            raise ValueError("Unsupported model type. Use 'v12' or 'v5'.")
        best_idx = get_new_batch_idx(random=False, al=al, project=project, name=name)
        shutil.rmtree(detection_dir, ignore_errors=True)  # Clean up detection directory after use

    # Select train and val indices from the best candidates
    train_val_idx = best_idx[:train_images + val_images] # Total indices to select for train and val
    shuffle(train_val_idx) # Shuffle the selected indices to ensure randomness
    train_idx = train_val_idx[:train_images] # First part for training
    val_idx = train_val_idx[train_images:train_images + val_images] # Second part for validation
    
    def copy_files(file_list, dst_dir):
        src_dir = os.path.join('dataset', dataset)
        for file in file_list:
            for file_type in ['.txt', '.jpg']:
                src = os.path.join(src_dir, f'{file}{file_type}')
                dst = os.path.join(dst_dir, f'{file}{file_type}')
                shutil.copy(src, dst)

    copy_files(train_idx, os.path.join('dataset', 'run', 'train'))
    copy_files(val_idx, os.path.join('dataset', 'run', 'val'))
