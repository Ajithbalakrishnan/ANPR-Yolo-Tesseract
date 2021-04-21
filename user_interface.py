import cv2
import time
import pytesseract
import numpy as np
from postprocess_anpr import *

tracker_id_1 = 0
tracker_id_2 = 0
glob_lp_frame = blank_image(250,150)
glob_lp = " "
counter = 0 
def tracker_process(lp,id,frameid):
    global counter
    global tracker_id_1
    global tracker_id_2
    global glob_lp 
    
    tracker_id_1 = id
    
    if tracker_id_1 == tracker_id_2:
        counter = counter + 1
    else:
        counter =0
        tracker_id_2 = tracker_id_1
    if counter >= CONFIDANCE_FRAMES:
        print("true")
        return lp

    else:
        global glob_lp_frame
        global glob_lp 
        return glob_lp_frame


def img_retrive(Lp_q):
    print("start 3rd process")

    while True:

        if Lp_q.empty == True:
            continue
        try:    
            obj = Lp_q.get(False)
            if len(obj.lp) == 0:
                feed = obj.frame
                lp_img = blank_image(250,150)
                lp_pred = "   "
                vehicle_type = "    "
                start_ui(feed_img=feed,lp_img=lp_img,predicted_number=lp_pred,vehicle_type = vehicle_type)
            else:
                feed = obj.frame
                lp_img = obj.lp
                lp_pred = obj.lp_pred
                track_id = obj.Tracker_ID
                frameid =obj.frame_id
                lp_image= tracker_process(lp_img,track_id,frameid)
                vehicle_type = obj.vehicle_type 
                start_ui(feed_img=feed,lp_img=lp_image,predicted_number=lp_pred,vehicle_type = vehicle_type)
        except Exception as e:
            #print(e)
            #print("Img Retrive process error")
            continue

       # predicted_number = pytesseract_(lp_img)

def start_ui(status=None,lp_img=None,feed_img=None,predicted_number=None,vehicle_type=None):

        background_image = blank_image(800,1000)
        feed_img_edit = cv2.cvtColor(feed_img, cv2.COLOR_RGB2RGBA).copy()
        LP_img_edit = cv2.cvtColor(lp_img, cv2.COLOR_RGB2RGBA).copy()
        img_1 = overlay_image_alpha(background_image,
                            feed_img[:, :, 0:3],
                            (80, 150),
                            feed_img_edit[:, :, 3] / 255.0)
        img_2 = overlay_image_alpha(img_1,
                            lp_img[:, :, 0:3],
                            (800, 150),
                            LP_img_edit[:, :, 3] / 255.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        type_of_vehicle = "vehicle_type : " + str(vehicle_type)
        
        
        display_num = "LP_Number: " +str(predicted_number) 

        headline = "ANPR System v1"
        cv2.putText(img_2, headline, (280,100), font, 1, (120, 120, 0), 3, cv2.LINE_AA)
        cv2.putText(img_2, type_of_vehicle, (750,300), font, .5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_2, display_num, (750,400), font, .5, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("image_",img_2)
        cv2.waitKey(100) 