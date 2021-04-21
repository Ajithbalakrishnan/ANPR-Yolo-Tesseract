import cv2
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import random
import numpy as np
import configparser
import matplotlib.pyplot as plt  
import multiprocessing
config = configparser.ConfigParser()
config.read("anpr_config.cfg")

CAM_URL = config["CAMERA_PARAMETERS"]["URL"]
CAM_TYPE = config["CAMERA_PARAMETERS"]["TYPE"]

YOLO_DETECTION_CONFIG_PATH = config["YOLO_LP_DETECTION"]["CONFIG_PATH"]
YOLO_DETECTION_WEIGHT_PATH = config["YOLO_LP_DETECTION"]["WEIGHT_PATH"]
YOLO_DETECTION_META_PATH = config["YOLO_LP_DETECTION"]["META_PATH"]
YOLO_DETECTION_THRESHOLD = int(config["ALGO_PARAMS"]["YOLO_LP_DETECTION_THRESHOLD"])
CONFIDANCE_FRAMES = int(config["ALGO_PARAMS"]["CONFIDANCE_FRAMES_LPD"])

YOLO_OCR_CONFIG_PATH = config["YOLO_LP_OCR"]["CONFIG_PATH"]
YOLO_OCR_WEIGHT_PATH = config["YOLO_LP_OCR"]["WEIGHT_PATH"]
YOLO_OCR_META_PATH = config["YOLO_LP_OCR"]["META_PATH"]

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax       

def crop_lp(frame,points,vehicle_type):
    x = points[1]
    y = points[0]
    h = points[2]
    w = points[3]
 
    xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
    crop = frame[xmin:xmax, ymin:ymax]
    return crop

def pytesseract_(img):
    img_cv = img
    # img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("img.jpg",img_rgb)
    # img = cv2.imread('img.jpg',0)
    
    # ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    # th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # cv2.imshow('threshold', th2)
    # cv2.waitKey(100) 
    # cv2.destroyWindow('threshold')

  #    img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)

    
    lp_num = pytesseract.image_to_string(img_cv)
    
    print("lp_num ",lp_num)
    return (lp_num)

def img_preprocess():
    img = cv2.imread('/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/290.jpeg')
    # for gamma in [0.1, 0.5, 1.2, 2.2]: 
    #     gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8') 
    #     cv2.imwrite('gamma_transformed'+str(gamma)+'.jpg', gamma_corrected) 
    cv2.imshow("img",img)
    # img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    stretch_near = cv2.resize(img, (780, 540),  
               interpolation = cv2.INTER_NEAREST) 
    cv2.imshow("img_stretch_near",stretch_near)
    cv2.waitKey(3000)
    # bitwise_not_img = cv2.bitwise_not(img)
    # cv2.imshow("bitwise_not",bitwise_not_img)
    
    # ret, mask = cv2.threshold(bitwise_not_img, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow("image_threshold",mask)

    # cv2.waitKey(3000)
    # pytesseract_(img)
    # digits = histogram_of_pixel_projection(img)
    
def blank_image(x,y):
    blank_img = 180 * np.ones(shape=[x, y, 3], dtype=np.uint8)
    return blank_img

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):

    x, y = pos

    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
    return img     

