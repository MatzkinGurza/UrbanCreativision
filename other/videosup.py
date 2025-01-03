import numpy as  np
from typing import Optional
import cv2
from sideseeing_tools import sideseeing
from ultralytics import YOLO
import supervision as sv
from ultralytics.solutions import ObjectCounter

#FUNCTIONS

def video_explore(vidpath: Optional[str]=None,
                   url: Optional[str]=None, 
                   set_fps: bool= False, 
                   vid_fps: Optional[int]=60, 
                   window_size: Optional[tuple[int,int]]=(600,450)):
    
    '''
    Set "set_fps" to True if vid_fps should be used as parameter. <p>
    Default value of "set_fps" is False, in this case, OpenCV will try to find the video's FPS by itself.<p>
    Default value of "vid_fps" is 60. <p>
    Press "q" to exit video window or click the X on the top right corner of the window.
    window_size = (x, y) = (width, height)
    '''

    try:
        if vidpath: #looks for vidpath to see if local video will be ran
            vid = cv2.VideoCapture(vidpath)
            if not vid.isOpened():
                raise IOError("Error: Could not open video file.")
            else:
                cv2.namedWindow("video", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("video", window_size[0], window_size[1])
                if set_fps: 
                    fps = vid_fps
                else:
                    fps = vid.get(cv2.CAP_PROP_FPS)
                    print(f"retrieved FPS: {fps}")
                if not fps:
                    print(f"Failed to get FPS for waitkey(delay) delay, {vid_fps}FPS will be assumed")
                    delay = (1000/vid_fps)
                else:
                    delay = int(1000/fps)
                    print(f"waitkey delay: {delay}")
                while True: #will showcase the video while q is not pressed and while ret is true
                    ret, frame = vid.read()
                    if not ret:
                        break
                    cv2.imshow("video", frame)
                    if (cv2.waitKey(delay) & 0xFF == ord('q')) or (cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1): #detecting exit thru ascii mask for q command or window being closed
                        print("Window closed by the user.")
                        break

        elif url:
            raise NotImplemented("Error: functionality for processing non local vidoes has not been implemented yet")
        
        else:
            raise Exception("Error: Neither video path nor url were found")
        
        print("Video Explore Execution was Successful")

    except Exception as e:
        print(f"{e}")
    finally:
        if vid and vid.isOpened():  #Ensure vid is not None and is opened before releasing
            vid.release()
        cv2.destroyAllWindows()
        return print("Video Explore Execution Ended")

def apply_YoloTracker(vidpath: str, 
                      yolo_model: Optional[str]="yolov8n.pt", 
                      set_fps: bool= False,
                      vid_fps: Optional[int]=60, 
                      url: Optional[str]=None,
                      window_size: Optional[tuple[int,int]]=(600,450)):
    
    '''
    Default model is yolo v11 nano (yolov8n.pt). <p>
    Set "set_fps" to True if vid_fps should be used as parameter. <p>
    Default value of "set_fps" is False, in this case, OpenCV will try to find the video's FPS by itself.<p>
    Default value of "vid_fps" is 60. <p>
    Press "q" to exit video window or click the X on the top right corner of the window.
    '''

    try:
        if vidpath: #looks for vidpath to see if local video will be ran
            vid = cv2.VideoCapture(vidpath)
            if not vid.isOpened():
                raise IOError("Error: Could not open video file.")
            else:
                model = YOLO(yolo_model)
                cv2.namedWindow("video", cv2.WINDOW_NORMAL) 
                cv2.resizeWindow("video",  window_size[0], window_size[1])
                if set_fps: 
                    fps = vid_fps
                else:
                    fps = vid.get(cv2.CAP_PROP_FPS)
                    print(f"retrieved FPS: {fps}")
                if not fps:
                    print(f"Failed to get FPS for waitkey(delay) delay, {vid_fps}FPS will be assumed")
                    delay = (1000/vid_fps)
                else:
                    delay = int(1000/fps)
                    print(f"waitkey delay: {delay}")
                while True: #will showcase the video while q is not pressed and while ret is true
                    ret, frame = vid.read()
                    if not ret:
                        break
                    #Apply Yolo model to frame in order to track elements
                    model_result = model.track(frame, persist=True)
                    #Draw results
                    frame_ = model_result[0].plot()
                    cv2.imshow("video", frame_)
                    if (cv2.waitKey(delay) & 0xFF == ord('q')) or (cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1): #detecting exit thru ascii mask for q command or window being closed
                        print("Window closed by the user.")
                        break

        elif url:
            raise NotImplemented("Error: functionality for processing non local vidoes has not been implemented yet")
        
        else:
            raise Exception("Error: Neither video path nor url were found")
        
        print("Apply_YoloTracker Execution was Successful")

    except Exception as e:
        print(f"{e}")
    finally:
        if vid and vid.isOpened():  #Ensure vid is not None and is opened before releasing
            vid.release()
        cv2.destroyAllWindows()
        return print("Apply_YoloTracker Execution Ended")
    
def apply_YoloLive(yolo_model: Optional[str]="yolov8n.pt"):
    
    '''
    Default model is yolo v11 nano (yolov8n.pt). <p>
    Press "q" to exit video window or click the X on the top right corner of the window.
    '''
    try:
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            raise IOError("Error: Could not open video from webcam")
        else:
            model = YOLO(yolo_model)
            cv2.namedWindow("video", cv2.WINDOW_NORMAL) #vidpath will, from now on, be used as window name string
            while True: #will showcase the video while q is not pressed and while ret is true
                ret, frame = vid.read()
                if not ret:
                    break
                #Apply Yolo model to frame in order to track elements
                model_result = model(frame)
                #Draw results
                frame_ = model_result[0].plot()
                cv2.imshow("video", frame_)
                if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1): #detecting exit thru ascii mask for q command or window being closed
                    print("Window closed by the user.")
                    break
        
        print("Apply_YoloLive Execution was Successful")

    except Exception as e:
        print(f"{e}")
    finally:
        if vid and vid.isOpened():  #Ensure vid is not None and is opened before releasing
            vid.release()
        cv2.destroyAllWindows()
        return print("Apply_YoloLive Execution Ended") 

def apply_YoloCounter(vidpath: str, 
                      counter_region: list[tuple[int,int]],
                      yolo_model: Optional[str]="yolov8n.pt", 
                      set_fps: bool= False,
                      vid_fps: Optional[int]=60, 
                      url: Optional[str]=None,
                      frame_size: Optional[tuple[int,int]]=None):
    
    '''
    Default model is yolo v8 nano (yolov8n.pt). <p>
    counter_regio should be a four tuple list, where each tuple is composed of two ints, in order to draw a rectangular region.<p>
    counter_region defines where to setup yolo counter area - might not be available for early versions of yolo.<p>
    Defined counter_region vertices are place in respect to frame_size.
    If no frame size is set, cv2 will decide frame size to be used.
    Set "set_fps" to True if vid_fps should be used as parameter. <p>
    Default value of "set_fps" is False, in this case, OpenCV will try to find the video's FPS by itself.<p>
    Default value of "vid_fps" is 60. <p>
    Press "q" to exit video window or click the X on the top right corner of the window.
    '''

    try:
        if vidpath: #looks for vidpath to see if local video will be ran
            vid = cv2.VideoCapture(vidpath)
            if not vid.isOpened():
                raise IOError("Error: Could not open video file.")
            else:
                cv2.namedWindow("video", cv2.WINDOW_NORMAL) #vidpath will, from now on, be used as window name string
                width, height, fps = (int(vid.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
                
                #Setting fps
                if set_fps: 
                    fps = vid_fps
                else:
                    print(f"retrieved FPS: {fps}")
                if not fps:
                    print(f"Failed to get FPS for waitkey(delay) delay, {vid_fps}FPS will be assumed")
                    delay = (1000/vid_fps)
                else:
                    delay = int(1000/fps)
                    print(f"waitkey delay: {delay}")

                #initializing object counter with show=False in order to display video through cv2
                counter = ObjectCounter(show=False, 
                                        region=counter_region, 
                                        model=yolo_model, 
                                        ) 

                while True: #will showcase the video while q is not pressed and while ret is true
                    ret, frame = vid.read()
                    if not ret:
                        break
                    frame = frame if not frame_size else cv2.resize(frame, frame_size)
                    #applying counter to frame
                    frame_=counter.count(frame)
                    #showing frame
                    cv2.imshow("video", frame_)
                    if (cv2.waitKey(delay) & 0xFF == ord('q')) or (cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1): #detecting exit thru ascii mask for q command or window being closed
                        print("Window closed by the user.")
                        break

        elif url:
            raise NotImplemented("Error: functionality for processing non local vidoes has not been implemented yet")
        
        else:
            raise Exception("Error: Neither video path nor url were found")
        
        print("Apply_YoloCounter Execution was Successful")

    except Exception as e:
        print(f"{e}")
    finally:
        if vid and vid.isOpened():  #Ensure vid is not None and is opened before releasing
            vid.release()
        cv2.destroyAllWindows()
        return print("Apply_Counter Execution Ended")
    

