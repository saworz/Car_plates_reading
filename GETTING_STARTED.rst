Detecting and reading car plates with YOLOv5 & easyOCR
=====================

Script uses YOLOv5 model trained on custom dataset to detect car plates.
Detected plates are cropped and passed to easyOCR to get readings.
There is also implemented simple object tracking to make sure that
the displayed solutions are not mixed on a video/camera device

How to run the code?
====================

1. Create virtual environment and install packages from requirements.txt

:Create env: `python3 -m venv /path/to/new/virtual/environment`

:Activate env: `source /path/to/new/virtual/environment`

:Install requirements: `pip install -r requirements.txt`

:Install torch: `py -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio===0.13.1 -f https://download.pytorch.org/whl/torch_stable.html`

:Run script: `py /path/to/main.py --epochs --lr --seed --batch --workers`

2. Clone yolov5 from github and install dependencies

:Clone yolo: `git clone https://github.com/ultralytics/yolov5`

:Dir in: `cd yolov5`

:Install dependencies: `pip install -r requirements.txt`

3. To train model prepare a directory with images and a directory with annotations which have names corresponding to images.

`!` Can run on GPU or CPU

Sample dataset:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection