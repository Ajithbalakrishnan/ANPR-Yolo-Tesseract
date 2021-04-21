import time
class Lp_Frame:
    def __init__(self, frame, frame_id, detection_list = None, time = time.time(), lp_img = None, vehicle_type = None, tracker_id = None ):
        self.frame = frame
        self.frame_id = frame_id
        if detection_list == None:
            self.detlist = []
            self.lp = []
            self.lp_pred = None
            self.vehicle_type = None
            self.Tracker_ID = None
        else:
            self.detlist = detection_list
            self.lp = lp_img
            self.lp_pred = []
            self.vehicle_type = vehicle_type
            self.Tracker_ID = tracker_id
        self.time = time

