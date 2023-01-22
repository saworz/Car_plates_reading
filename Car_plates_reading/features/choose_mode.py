from features.cv2_handling import cv2_handling
from features.image_mode import image_mode

import torch
import sys
import easyocr

def choose_mode(model: torch.nn.Module):

    reader = easyocr.Reader(['en'])
    
    while True:
        print("Choose mode: \n1. Device camera\n2. Video path\n3. Image path")
        mode = input("Input mode: ")

        if mode == '1':
            cv2_handling(0, model, reader)
        elif mode == '2':
            try:
                video_path = input("Enter path to the video file: ")
                cv2_handling(video_path, model, reader)
            except:
                print("Incorrect path...") 
        elif mode == '3':
            try:
                photo_path = input("Enter path to the image file: ")
                image_mode(photo_path, model, reader)
            except:
                print("Incorrect path...")
        elif mode.lower() == 'q':
            sys.exit()
        else:
            print("Incorrect mode, try again...")