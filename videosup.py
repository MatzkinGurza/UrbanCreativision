import numpy as  np
from typing import Optional
import cv2
from sideseeing_tools import sideseeing
from ultralytics import YOLO
import supervision as sv


def video_explore(vidpath: Optional[str]=None, url: Optional[str]=None, set_fps: bool= False, vid_fps: Optional[int]=60):
    
    '''
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
                cv2.resizeWindow("video", 600, 450)
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


def apply_YoloTracker(vidpath: str, yolo_model: Optional[str]="yolov8n.pt", set_fps: bool= False, vid_fps: Optional[int]=60, url: Optional[str]=None):
    
    '''
    Default model is yolo v8 nano (yolov8n.pt). <p>
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
                cv2.namedWindow("video", cv2.WINDOW_NORMAL) #vidpath will, from now on, be used as window name string
                cv2.resizeWindow("video", 600, 450)
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
    
def main(yolo_model: Optional[str]="yolov8n.pt"):
    
    try:
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            raise IOError("Error: Could not open video from webcam")
        else:
            model = YOLO(yolo_model)
            cv2.namedWindow("video", cv2.WINDOW_NORMAL) #vidpath will, from now on, be used as window name string
            cv2.resizeWindow("video", 600, 450)
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
        
        print("Video Explore Execution was Successful")

    except Exception as e:
        print(f"{e}")
    finally:
        if vid and vid.isOpened():  #Ensure vid is not None and is opened before releasing
            vid.release()
        cv2.destroyAllWindows()
        return print("Video Explore Execution Ended")    
    
#test module
if __name__ == "__main__":
    main()