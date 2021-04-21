import os
import cv2 
import sys
import time
import math
import random
sys.path.append('/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/ANPR_v2')
import darknet
import subprocess
import numpy as np
from ctypes import *
import multiprocessing
from postprocess_anpr import *
print("number of cpu : ", multiprocessing.cpu_count())

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def crop_lp(frame,points):
    x = points[1]
    y = points[0]
    h = points[2]
    w = points[3]
 
    xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
    crop = frame[xmin:xmax, ymin:ymax]
    return crop

netMain = None
metaMain = None
altNames = None

def Lp_recognition(Vehicle_q,Lp_q):

    global metaMain, netMain, altNames
    configPath = str(YOLO_OCR_CONFIG_PATH)
    weightPath = str(YOLO_OCR_WEIGHT_PATH)
    metaPath = str(YOLO_OCR_META_PATH)

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    
    print("Starting the YOLO loop...")
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
        
    while True:
        try: 

            if Lp_q.full():
                Lp_q.get(False)

            if Vehicle_q.empty == True :
                continue

            obj = Vehicle_q.get(False)
            if len(obj.lp) == 0:
              
                Lp_q.put(obj)
                continue

            frame_read = obj.lp
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
        
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.65)
            
            size_of = len(detections)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow("Frame", image)
            # cv2.waitKey(1)
            time.sleep(.050)
            x=0
            lp_predicted = ""
            for lp_num in range(len(detections)):
                    
                    try:
                        value = detections[int(lp_num)][0]                    
                        lp_predicted = str(lp_predicted) + str(value,'utf-8')

                    except Exception as e:                 
                        continue

            #print("lp_predicted : ",lp_predicted)
            obj.lp_pred = lp_predicted
            Lp_q.put(obj)

        except Exception as e:
                        continue

    
# if __name__ == "__main__":
#     Lp_recognition()
