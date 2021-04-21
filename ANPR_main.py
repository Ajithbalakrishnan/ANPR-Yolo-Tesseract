try:
    os.system("killall -9 python")
    time.sleep(100)
    print("old python process killed ")

except:
    print("No running python process ")
    pass

from ANPR_lp_detection import *
from ANPR_lp_recognition import *
from postprocess_anpr import * 
from user_interface import *
import multiprocessing

def main():
    print("main start")
    
    Lp_q = multiprocessing.Queue(maxsize = 10)  

    Vehicle_q = multiprocessing.Queue(maxsize = 10)

    NP_Detection = multiprocessing.Process(target= anpr_detection, args=(Lp_q,Vehicle_q,))
    
    NP_Recognision = multiprocessing.Process(target= Lp_recognition, args=(Vehicle_q,Lp_q))

    ui_start = multiprocessing.Process(target = img_retrive, args=(Lp_q,))

    NP_Detection.start()

    NP_Recognision.start()

    ui_start.start()

    print("main end")

if __name__ == "__main__":
    main()