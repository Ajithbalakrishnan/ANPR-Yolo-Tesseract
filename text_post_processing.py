import cv2
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import random
import numpy as np
import matplotlib.pyplot as plt  
from cropyble import Cropyble
import keras_ocr
image__ = '/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/lp/55.jpeg'
# H 10 7 13 30 0
# R 13 7 18 30 0
# O 22 7 39 30 0
# V 41 7 45 30 0
# A 50 7 52 30 0
# U 52 7 68 31 0
# 0 75 8 84 31 0
# 0 84 8 90 31 0
# 7 90 8 92 31 0
# 2 96 9 110 32 0
def image_to_boxes():
    img = cv2.imread(image__)
 #   img = img_stretch (img,200,70)
    img_2 = cv2.resize(img, None, fx=2, fy=2)
    cv2.imshow("img_2",img_2)
    h, w, _ = img_2.shape
    
    ret,thresh1 = cv2.threshold(img_2,50,255,cv2.THRESH_BINARY)
    boxes = pytesseract.image_to_boxes(thresh1)
    cv2.imshow("thresh1",thresh1)

    pytesseract_(thresh1)
    pytesseract_(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        img_k = cv2.rectangle(img_2, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    cv2.imshow("Annotations", img_k)
    cv2.waitKey(0)

def pytesseract_(img):
    img_cv = img
#    custom_oem_psm_config = r'--psm 10 --oem 3'
#    lp_num = pytesseract.image_to_string(img_cv, lang='eng', config=custom_oem_psm_config)
    lp_num = pytesseract.image_to_string(img_cv)
#    print("pytesseract.image_to_boxes",pytesseract.image_to_boxes(Image.open(image__)))
#    hocr = pytesseract.image_to_pdf_or_hocr('/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/290.jpeg', extension='hocr')
#    print("hocr",hocr)

    print("lp_num ",lp_num)
    return (lp_num)

def img_thresholding(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    
    thresh2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    ret2,thresh3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return(thresh1,thresh2,thresh3)

def img_bitwise_not(img):
    bitwise_not_img = cv2.bitwise_not(img)
    return (bitwise_not_img)

def img_stretch(img,x,y):
    stretch_near = cv2.resize(img, (int(x), int(y)),interpolation = cv2.INTER_NEAREST) 
    return(stretch_near)
def ocr_keras(img):
    pipeline = keras_ocr.pipeline.Pipeline()

#    detector = keras_ocr.detection.Detector()
    image = keras_ocr.tools.read(img)

    prediction_groups = pipeline.recognize(image)
    fig, axs = plt.subplots(nrows=len(image), figsize=(20, 20))
    for ax, image, predictions in zip(axs, image, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
#    boxes = detector.detect(images=[image])[0]
#    print(boxes)
def main():
    img = cv2.imread('/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/290.jpeg')
    ocr_keras(img)

    pytesseract_(img)
    cv2.imshow("img",img)
    cv2.waitKey(3000)

if __name__ == "__main__":
    main()