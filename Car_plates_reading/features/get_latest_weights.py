import glob
import os

def get_weights(yolo_dir: str) -> str:
    list_of_files = glob.glob(yolo_dir + '/runs/train/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_weight_path = latest_file + '/weights/best.pt'
    return latest_weight_path