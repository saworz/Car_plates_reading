import torch
import numpy as np
import cv2
from PIL import Image
import easyocr

def image_mode(image_path: str, model: torch.nn.Module, reader: easyocr.Reader) -> None:

    results = model(image_path)
    detections=np.squeeze(results.render())

    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    image = cv2.imread(image_path)
    width, height = image.shape[1], image.shape[0]

    print(f'Photo width,height: {width},{height}. Detected plates: {len(labels)}')

    for i in range(len(labels)):
        row = coordinates[i]
        if row[4] >= 0.6:
            x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
            plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
            ocr_result = reader.readtext((plate_crop), paragraph="True", min_size=120, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            text=ocr_result[0][1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6) ## BBox
            cv2.putText(image, f"{text}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            
            print(f'Detection: {i+1}. YOLOv5 prob: {row[4]:.2f}, easyOCR results: {ocr_result}')

    image = Image.fromarray(np.uint8(np.squeeze(image[...,::-1])))
    image.show()