import os
from random import shuffle

def get_split(train_size=0.6, val_size=0.2, test_size=0.2, dataset_size=200):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
    test_indices = [int(file.split('.')[0]) for file in os.listdir(os.path.join("dataset/test")) if file != "classes.txt"]
    indices = list(range(dataset_size))
    indices = [idx for idx in indices if idx not in test_indices]
    shuffle(indices)
    n_train = int(train_size * dataset_size)
    n_val = int(val_size * dataset_size)
    return indices[:n_train], indices[n_train:]

def rm_from_dir():
    base_path = os.path.join(dir, 'dataset/run')
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(base_path, split)
        for file in os.listdir(split_path):
            file_path = os.path.join(split_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def get_avg_conf(file_path, mode='confidence'):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            if mode == 'count':
                return len(lines)
            total_conf = sum(float(line.strip().split()[5]) for line in lines)
            return total_conf / len(lines) if lines else 0
    except Exception:
        return 0 if mode in ['confidence', 'count'] else 1
    
    
def avg_conf(mode='RANDOM', AL_mode = 'confidence', project='batch', name='name_batch'):
    train_dir = os.path.join(dir, 'dataset/run/train')
    val_dir = os.path.join(dir, "dataset/run/val")
    train = [file for file in os.listdir(train_dir) if file.endswith('.jpg')]
    val = [file for file in os.listdir(val_dir) if file.endswith('.jpg')]
    
    train_val_images_dir = os.path.join(dir, 'dataset/train_val_images')
    to_eval = [file[:-4] for file in os.listdir(train_val_images_dir) if file not in train and file not in val]
    
    if mode not in ['AL', 'APPROACH']:
        return to_eval
    
    path = os.path.join(dir, project, name, 'labels')
    evals = [(eval_, get_avg_conf(os.path.join(path, f'{eval_}.txt'), mode=mode)) for eval_ in to_eval]
    evals.sort(key=lambda x: x[1])
    
    return [file[0] for file in evals]


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


def split_dataset(dataset="improved", train_split=[], val_split=[], test_split=[]):
    rm_from_dir()
    
    def copy_files(file_list, split):
        split_dir = os.path.join(dir, 'dataset/run', split)
        for file in file_list:
            for file_type in ['.jpg', '.txt']:
                src = os.path.join(dir, 'dataset', dataset if split != 'test' else 'improved', f'{file}{file_type}')
                dst = os.path.join(split_dir, f'{file}{file_type}')
                shutil.copy(src, dst)
    
    copy_files(train_split, 'train')
    copy_files(val_split, 'val')
    copy_files(test_split, 'test')