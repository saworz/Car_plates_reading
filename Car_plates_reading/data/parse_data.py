import argparse

def train_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="16", type=str, help='batch size')
    parser.add_argument("--epochs", default="1", type=str, help='amount of epochs')
    parser.add_argument("--workers", default="2", type=str, help='number of workers')
    parser.add_argument("--img_size", default="640", type=str, help='image size')
    opt = parser.parse_args()
    return opt