from features.plate_tracking import get_plates_xy
from features.plate_tracking import detect_text
from features.plate_tracking import sort_detections
from features.plate_tracking import delete_old_labels
from features.plate_tracking import overwrite_plates_data

import cv2
import torch
import numpy as np
import easyocr
from typing import Union

def cv2_handling(mode: Union[int, str], model: torch.nn.Module, reader: easyocr.Reader):

    cap = cv2.VideoCapture(mode)

    plates_data = [['None', [0,0], 0] for n in range(5)]
    count_empty_labels=[0]*5

    assert cap.isOpened()
    while(cap.isOpened()):
        ret, frame = cap.read()
        assert not isinstance(frame,type(None)), 'frame not found'
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  
        results = model(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        width, height = frame.shape[1], frame.shape[0]
        
        detections=[['None', [0,0], 0] for n in range(5)]
        i=0 
        
        
        ## Read all detected plates per each frame and save them to >>detections<<
        while i < len(labels):    
            row = coordinates[i]
            ## 3. Crop detections and pass them to the easyOCR
            ocr_result, x1, y1=get_plates_xy(frame, labels, row, width, height, reader)  
            
            ## 4. Get reading for the each frame
            detections=detect_text(i, row, x1, y1, ocr_result, detections, 0.5)
            i+=1    
        i=0
        
        ## 5. Do some tracking and data managing for better results
        ## If we get multiple detections in one frame easyOCR mixes them every few frames, so here we make sure that they are saved according to the \
        ## detections' coordinates. Then we delete data about plates that dissapeared for more than >>frames_to_reset<< frames. And finally we overwrite \
        ## the predictions (regarding to the probability of easyOCR detections - if new predcition has less p% than the previous one, we skip it.)
        
        ## Sort detections 
        detections=sort_detections(detections, plates_data)
        
        ## Delete data about plates that dissapeared from frame
        plates_data, count_empty_labels=delete_old_labels(detections, count_empty_labels, plates_data, 3)
                
        ## Overwrite data and print text predictions over the boxes
        while i < len(labels):
            plates_data=overwrite_plates_data(i, detections, plates_data, 7)
            cv2.putText(frame, f"{plates_data[i][0]}", (plates_data[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            i+=1

        #cv2.imshow('YOLO', np.squeeze(results.render()))
        cv2.imshow('YOLO', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()