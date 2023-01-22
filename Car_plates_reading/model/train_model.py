from data.get_annotations import get_annotations
from data.get_labels import get_labels
from data.split_data import split_data
from data.create_yaml import create_yaml
from features.get_latest_weights import get_weights
from data.parse_data import train_options

import torch
import subprocess

def train_model() -> torch.nn.Module:

    opt = train_options()

    print("""Input path to the directory that contains folder named "annotations" with corresponding .xml files and folder "images" with .png/.jpg/.jpeg files""")
    dir_path = input("Enter dir path: ")
    yolo_dir = input("Enter path to yolov5 directory")

    classes = ['plate']

    df = get_annotations(dir_path, classes)
    df = get_labels(dir_path, df)
    split_data(dir_path, yolo_dir)

    create_yaml(yolo_dir, classes)

    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Working on a device: {device}")

    train_dir = yolo_dir + "/train.py"
    subprocess.run(["py", train_dir, "--workers", opt.workers, "--img", opt.img_size, "--batch", opt.batch,
    "--epochs", opt.epochs, "--data", yolo_dir + "\data\plates.yaml", "--weights", "yolov5s.pt", "--device",  "0", "--cache"])

    weights = get_weights(yolo_dir)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = weights, force_reload=True)

    return model