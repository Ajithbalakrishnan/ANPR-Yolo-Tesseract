import string  
import os
import os.path
import re
import shutil
from PIL import Image
from shutil import copyfile
import argparse
import glob
import math
import random
import time 
import cv2

def apply_boarder(img):
    borderType = cv2.BORDER_CONSTANT

    top = int(210)
    bottom = top
    left = int(210)
    right = left
    value = [255,255,255]
    dst = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)
    return dst
def iterate_dir(source, dest):
    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    source= "/home/ajithbalakrishnan/vijnalabs/My_Learning/FL/numberplate_recognition/datasets/binary_img_dataset/"
    dest =  "/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/new_dataset/"

    for alp in range(len(alphabets)):
        print("alphabet:",alphabets[alp])

        source_ = source + str(alphabets[alp])+'/'
    
        images = [f for f in os.listdir(source_)
                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

        num_img = len(images)
        print("Number of image files",num_img)
    
        p = 0

        path = os.path.join(dest, alphabets[alp]) 
        print("path:",path)
        os.mkdir(path)  

        for z in range(num_img):
            idx = random.randint(0,len(images)-1)
            filename = images[idx]

            image = cv2.imread(str(source_ + filename))

            bitwise_not_img = cv2.bitwise_not(image)

            final_img = apply_boarder(bitwise_not_img)


            new_filename = "inv"+str(filename)
            destination = str(path +'/'+ new_filename) 
            print("destination",destination)
            cv2.imwrite(destination,final_img)
            images.remove(filename)

            time.sleep(.01)
            print("compleated:",str(p))
            p = p+1



def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        #default=os.getcwd()
        default= "/home/ajithbalakrishnan/vijnalabs/My_Learning/FL/numberplate_recognition/datasets/binary_img_dataset/Z/"
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default="/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/sample/Z/"
    )

    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir)  

if __name__ == '__main__':
    main()