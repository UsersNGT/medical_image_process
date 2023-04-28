import pydicom
import cv2
import numpy as np

def singledicom2jpg(path, win_width, win_level, save_path):
    ds = pydicom.dcmread(path)
    data = ds.pixel_array
    np.clip(data, win_level - win_width/2 , win_level + win_width/2)
    data = data / float(win_width)
    data = data * 255.0
    # print(data.shape,)
    cv2.imwrite(save_path, data)
