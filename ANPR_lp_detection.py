import os
import cv2 
import sys
import time
import math
import random
import pickle
import datetime
sys.path.append('/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/ANPR_v2')
import darknet
import subprocess
import numpy as np
from ctypes import *
import multiprocessing
from postprocess_anpr import *
from lp_data_structure import Lp_Frame
from pyimagesearch.centroidtracker import CentroidTracker
total_class_dict = {'car':1,'truck':2,'bus':3,'motorcycle':4,'auto':5,'carLP':6,'truckLP':7,'busLP':8,'motorcycleLP':9,'autoLP':10} 
lp_class_dict = {'carLP':'Car','truckLP':'Truck','busLP':'Bus','motorcycleLP':'Two Wheeler','autoLP':'Three Wheeler'}
lp_class_list = ['carLP','truckLP','busLP','motorcycleLP','autoLP']
print("number of cpu : ", multiprocessing.cpu_count())
frameid  = 0

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

ct = CentroidTracker()
w_size = 640
h_size = 640

def lp_centroid(points,frame):
    if(len(points)!=0):
        rects = []
        x1, y1, w_size, h_size = points
        x_start = round(x1 - (w_size/2))
        y_start = round(y1 - (h_size/2))
        x_end = round(x_start + w_size)
        y_end = round(y_start + h_size)
        data = [x_start, y_start, x_end, y_end] * np.array([1,1,1,1])
        rects.append(data)
    else:
        rects = []
    objects = ct.update(rects)
    
    
    for (objectID, centroid) in objects.items():

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    # cv2.imshow("Frame", frame)
    # cv2.waitKey(1)
    return objects 

netMain = None
metaMain = None
altNames = None

def anpr_detection(Lp_q,Vehicle_q):

    global metaMain, netMain, altNames
    configPath = str(YOLO_DETECTION_CONFIG_PATH)
    weightPath = str(YOLO_DETECTION_WEIGHT_PATH)
    metaPath = str(YOLO_DETECTION_META_PATH)

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
            "ascii"), weightPath.encode("ascii"), 0, 1) 
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
    cap = cv2.VideoCapture(str(CAM_URL))
#    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    print("Starting the YOLO loop...")

    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_resized_ = cv2.resize(frame_read, (darknet.network_width(netMain),darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR) 
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.90)
        
        time.sleep(.075)
        global frameid 
        frameid =  frameid + 1
        detect_size = len(detections)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print("type detections : ",type(detections))

        try:
            if Vehicle_q.full():
                Vehicle_q.get(False)

            if detect_size == 0:
                Vehicle_q.put(Lp_Frame(frame=frame_resized_,frame_id = str(frameid) ,time=time.time()),False)
                lp_centroid(detections,frame_resized)
            else:
                for lp_num in range(len(detections)):
                    search_key =str(detections[int(lp_num)][0],'utf-8')
                    if (search_key in lp_class_list):
                        res = [val for key, val in lp_class_dict.items() if search_key in key] 
                        lp_points= detections[int(lp_num)][2]

                        crop_lp_img = crop_lp(frame_resized_,lp_points)
                        objects = lp_centroid(lp_points,frame_resized)
                        for (objectID, centroid) in objects.items():
                            text = "{}".format(objectID)
                            print("tracker_id : ",text)

                        filename = "/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/ANPR/images/"+str(datetime.datetime.now())+'.jpg'
                        cv2.imwrite(filename, crop_lp_img)
                        Vehicle_q.put(Lp_Frame(frame=frame_resized_,frame_id = str(frameid), detection_list=detections,lp_img=crop_lp_img, tracker_id = str(text), vehicle_type= str(res[0]),time=time.time()),False)
                        
                    else:
                        Vehicle_q.put(Lp_Frame(frame=frame_resized_,frame_id = str(frameid), time=time.time()),False)
                        null_list = []
                        lp_centroid(null_list,frame_resized)

            # x = Vehicle_q.get(False)

            # if (len(x.detlist) == 0):
            #     print("no detections buddy")
            # else:
            #     print("Detections: ", x.detlist)
            #     print("Vehicle Type: ",x.vehicle_type)
            #     cv2.imshow("lp",x.lp)
            # cv2.imshow("frame",x.frame)
            # cv2.waitKey(1)
            
            # p = Vehicle_q.get(False)

            # try:
            #     if Lp_q.full():
            #         Lp_q.get(False)

            #     if (len(p.detlist) == 0):
            #         print("No detections buddy")
            #         Lp_q.put(p,False)
            #     else:
            #         p.lp_pred = "Sample Lp"
            #         Lp_q.put(p,False)

            # except Exception as e:
            #     print(e)
            #     print("Error while loading datas in Lp_q")

            # y = Lp_q.get(False)
            # if(len(y.detlist)==0):
            #     print("No detections")
            # else:
            #     print("Detections: ", y.detlist)
            #     print("Vehicle Type: ",y.vehicle_type)
            #     print("Lp Number: ",y.lp_pred)
            #     cv2.imshow("lp",y.lp)
            # cv2.imshow("frame",y.frame)
            # cv2.waitKey(1)

        except Exception as e:
            print("Error while Loading Image to vehicle queue")
            continue

    #cv2.destroyAllWindows()
    cap.release()
    out.release()
    
# if __name__ == "__main__":
#     anpr_detection()
