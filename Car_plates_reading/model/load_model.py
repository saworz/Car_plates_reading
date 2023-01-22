import torch
import sys

def load_model() -> torch.nn.Module:
    
    while True:
        weights = input("Enter path to existing weights: ")
        
        if weights.lower() == "q":
            sys.exit()

        try: 
            model = torch.hub.load('ultralytics/yolov5', 'custom', path = weights, force_reload=True)
            break
        except:
            print("Incorrect path, try again...")

    return model