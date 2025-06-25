import os
from random import sample, shuffle
import shutil

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

def get_new_batch_idx(random=False, al='confidence', project='.temp_bacth', name='batch'):
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
    available_idx = [idx for idx in train_val_idx if idx not in test_idx]

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


def new_batch(weights="''", project="batch", name="name_batch", mode="RANDOM", train_size=0.05, val_size=0.05, split_random=True, dataset="improved"):
    if mode in ["AL", "APPROACH"]:
        detect(weights=weights, project=project, name=name, source="dataset/train_val_images/")
    
    batch = avg_conf(mode=mode, project=project, name=name)
    
    if mode not in ["AL", "APPROACH"]:
        batch = sample(batch, int(DATASET_SIZE * (train_size + val_size)))
        train = batch[:int(DATASET_SIZE * train_size)]
        val = batch[int(DATASET_SIZE * train_size):int(DATASET_SIZE * train_size) + int(DATASET_SIZE * val_size)]
    else:
        batch = batch[:int(DATASET_SIZE * (train_size + val_size))]
        if split_random:
            train = sample(batch, int(DATASET_SIZE * train_size))
            val = [im for im in batch if im not in train]
        else:
            train = batch[:int(DATASET_SIZE * train_size)]
            val = batch[int(DATASET_SIZE * train_size):int(DATASET_SIZE * train_size) + int(DATASET_SIZE * val_size)]
    
    train_dst_dir = os.path.join(dir, "dataset/run/train")
    val_dst_dir = os.path.join(dir, "dataset/run/val")
    official_dir = os.path.join(dir, "dataset/official", dataset)
    
    for file in train:
        for file_type in ['.txt', '.jpg']:
            src = os.path.join(official_dir, file + file_type)
            dst = os.path.join(train_dst_dir, file + file_type)
            shutil.copy(src, dst)
    
    for file in val:
        for file_type in ['.txt', '.jpg']:
            src = os.path.join(official_dir, file + file_type)
            dst = os.path.join(val_dst_dir, file + file_type)
            shutil.copy(src, dst)
