Detecting and reading car plates with YOLOv5 & easyOCR
==============================

Script uses YOLOv5 model trained on custom dataset to detect car plates.
Detected plates are cropped and passed to easyOCR to get readings.
There is also implemented simple object tracking to make sure that
the displayed solutions are not mixed on a video/camera device

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── Car_plates_reading      <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data                     <- Scripts to handle data
    │   │   └── create_yaml.py       <- Creates .yaml file required for yolo training 
    │   │   └── get_annotations.py   <- Read annotations as dataframe
    │   │   └── get_labels.py        <- Read labels and add to dataframe
    │   │   └── parse_data.py        <- Data parser
    │   │   └── split_data.py        <- Splits data into training and validation dirs
    │   │
    │   ├── features                     <- Scripts to turn raw data into features for modeling
    │   │   └── choose_loading.py        <- Used to choose between training and loading model
    │   │   └── choose_mode.py           <- Choose mode to use
    │   │   └── cv2_handling.py          <- Cv2 script to detect pkates on a video/camera
    │   │   └── get_latest_weights.py    <- Load latest weights
    │   │   └── image_mode.py            <- Script to detect plates on an image
    │   │   └── plate_tracking.py        <- Simple plates tracking for video/camera mode
    │   │
    │   ├── model                  <- Scripts to train/load model
    │   │   │── train_model.py     <- Train new model               
    │   │   ├── load_model.py      <- Load existing weights to the model
    │   │
    └── tox.ini            <- tox file with settings for running tox

--------

Sample dataset:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
