import sys
from random import randint
import cv2 as cv
def main():
    
    borderType = cv.BORDER_CONSTANT
    window_name = "copyMakeBorder Demo"
    
    src = cv.imread("/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/290.jpeg",0)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: copy_make_border.py [image_name -- default lena.jpg] \n')
        return -1
    
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    
#    top = int(0.05 * src.shape[0])  # shape[0] = rows
    top = int(210)
    bottom = top
#    left = int(0.05 * src.shape[1])  # shape[1] = cols
    left = int(210)
    right = left
    
    th3 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
  #  th1 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)   
    value = [255,255,255]
    dst = cv.copyMakeBorder(th3, top, bottom, left, right, borderType, None, value)
    #
    cv.imshow(window_name, dst)
        
    c = cv.waitKey(1000)
    cv.imwrite("/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/new_stretched.jpg", dst)
    if c == 27:
        exit
        
    return 0
if __name__ == "__main__":
    main()
