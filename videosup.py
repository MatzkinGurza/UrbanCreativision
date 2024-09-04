import numpy as  np
from typing import Optional
import cv2

def video_explore(vidpath: Optional[str]=None, url: Optional[str]=None, set_fps: bool= False, vid_fps: Optional[int]=60):
    
    '''
    set "set_fps" to True if vid_fps should be used as parameter. <p>
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