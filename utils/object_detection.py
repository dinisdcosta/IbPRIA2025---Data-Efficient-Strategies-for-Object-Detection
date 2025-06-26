import subprocess
import os
from ultralytics import YOLO

def train_yolov5(img=1280, epochs=50, data='dataset/data.yaml', name='new_train', project='train', weights="''"):
    dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "python", os.path.join('yolov5', 'train.py'),
        "--cache", "ram", "--batch", "-1", "--epochs", str(epochs), "--img", str(img),
        "--data", data,
        "--name", name,
        "--project", f"{project}/train",
        "--cfg", "yolov5s.yaml", "--weights", weights,
        "--device", "0"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def detect_yolov5(img=1280, source='imgs/', name='new_detect', project='detect', weights='best.pt'):
    dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "python", os.path.join('yolov5', 'detect.py'),
        "--save-txt", "--save-conf",
        "--source", f"{dir}/{source}",
        "--weights", weights,
        "--img", str(img),
        "--name", name,
        "--project", f"{project}/detect",
        "--device", "0"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def test_yolov5(img=1280, data='dataset/data.yaml', name='new_test', project='test', weights="''"):
    dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "python", os.path.join('yolov5', 'val.py'),
        "--data", f"{dir}/{data}",
        "--weights", weights,
        "--img", str(img), "--task", "test",
        "--name", name,
        "--project", f"{project}/test",
        "--device", "0", "--exist-ok"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def train_yolo(img=1280, epochs=50, data='dataset/data.yaml', name='new_train', project='train', weights="yolo11n.pt", batch=4):
    model = YOLO(weights)
    results = model.train(data=data, epochs=epochs, imgsz=img, name=name, project=project, batch= batch)
    return results

def detect_yolo(img=1280, source='imgs/', name='detect', project='.temp/detect', weights='yolo11n.pt'):
    model = YOLO(weights)
    results = model(source=source, imgsz=img, name=name, project=project, save=True, save_txt=True, save_conf=True)
    return results

def test_yolo(img=1280, data='dataset/data.yaml', name='test', project='.temp/test', weights="yolo11n.pt"):
    model = YOLO(weights)
    results = model.val(data=data, imgsz=img, name=name, project=project, task="test", exist_ok=True)
    return results