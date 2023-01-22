import sys
from model.train_model import train_model
from model.load_model import load_model

def choose_loading():

    print("---Insert Q to leave---")

    while True:
        print("Do you want to train new yolo model or load existing weights?")
        chosen_option = input("Enter T for train or L for load: ")
            
        if chosen_option.lower() == "t":
            model = train_model()
            break
        elif chosen_option.lower() == "l":
            model = load_model()
            break
        elif chosen_option.lower() == "q":
            sys.exit()
        else: 
            print("Incorrect input, try again")

    return model