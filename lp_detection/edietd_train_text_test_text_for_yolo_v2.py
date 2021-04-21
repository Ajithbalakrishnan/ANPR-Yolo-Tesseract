# This code is to create train.text,test.text files for yolo
#python word_edit_2.py -i /home/ajith/Downloads/car-tank/2-2 -o /home/ajith/Desktop/sample  

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

def iterate_dir(source, dest,file_name_,ratio_):
    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    source= "/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/new_dataset/"
    dest = source
    k= open((os.path.join(dest, str("train"))),"w+")
    t= open((os.path.join(dest, str(file_name_))),"w+")
    for alp in range(len(alphabets)):
        print("alphabet:",alphabets[alp])

        source_ = source + str(alphabets[alp])+'/'
       # dest = source_
    
        images = [f for f in os.listdir(source_)
                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

        text_files = [f for f in os.listdir(source_)
                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.txt)$', f)]

        print("Total Number of image files",len(images))
        num_test_img = int(ratio_*int(len(images)))
        print("Number of test image files",num_test_img)

        num_train_img = int(len(images))-int(ratio_*int(len(images)))
        print("Number of train image files",num_train_img)
        
        num_text = len(text_files)
        print("number of text files:", num_text)
        
        p = 0
    
    
    
        for z in range(num_train_img):
            idx = random.randint(0, len(images)-1)
            filename = images[idx]

        # text_filename = os.path.splitext(train)[0]+'.txt'
            
            k.writelines("data/obj/"+str(alphabets[alp])+'/'+str(filename))
            images.remove(filename)
            time.sleep(.01)
            k.write("\n")
            time.sleep(.01)
            print("compleated:",str(p))
            p = p+1
       

        p = 0
    
   
        for z in range(num_test_img):
            idx = random.randint(0, len(images)-1)
            filename = images[idx]

        #    text_filename = os.path.splitext(filename)[0]+'.txt'
            
            t.writelines("data/obj/"+str(alphabets[alp])+'/'+str(filename))
            
            images.remove(filename)
            time.sleep(.01)
            t.write("\n")
            time.sleep(.01)
            print("compleated:",str(p))
            p = p+1
    t.close()
    k.close()

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default="/home/ajithbalakrishnan/vijnalabs/My_Learning/my_workspace/darknet/lp_detection/new_dataset/"
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=.1,
        type=float
    )
    parser.add_argument(
        '-f', '--file_name',
        help='output file name '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default="test"
    )

    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.file_name,args.ratio)  

if __name__ == '__main__':
    main()