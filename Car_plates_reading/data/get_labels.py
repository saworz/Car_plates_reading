import pandas as pd
from pathlib import Path

def get_labels(path: str, df: pd.DataFrame) -> pd.DataFrame:
    x_pos = []
    y_pos = []
    frame_width = []
    frame_height = []

    labels_path = path + r"/labels/"

    Path(labels_path).mkdir(parents=True, exist_ok=True)
    save_type = 'w'

    for i, row in enumerate(df.iloc):
        current_filename = str(row.file[:-4])
        
        width, height, xmin, ymin, xmax, ymax = list(df.iloc[i][-6:])
        
        x=(xmin+xmax)/2/width
        y=(ymin+ymax)/2/height
        width=(xmax-xmin)/width
        height=(ymax-ymin)/height
        
        x_pos.append(x)
        y_pos.append(y)
        frame_width.append(width)
        frame_height.append(height)
        
        txt =  '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'
        
        if i > 0:
            previous_filename = str(df.file[i-1][:-4])
            save_type='a+' if current_filename == previous_filename else 'w'
        
        
        with open(labels_path + str(row.file[:-4]) +'.txt', save_type) as f:
            f.write(txt)
            
            
    df['x_pos']=x_pos
    df['y_pos']=y_pos
    df['frame_width']=frame_width
    df['frame_height']=frame_height

    return df