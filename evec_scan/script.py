# IMPORTS

import os
import glob
import random
from typing import Literal, Optional
import ffmpeg
import imageio
import pandas as pd

        

# FUNCTIONS

def find_files(root_directory: str, extensions: tuple[str]=('*.mp4')) -> list:
    '''
    extensions might be: '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm'
    '''
    # Use glob to recursively find all video files
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(root_directory, '**', ext), recursive=True))
    return video_files

def instance_build(video_files: list[str], build_id: Optional[str] = '') -> list:
    instance_counter = 0  # Counter for instance folders
    video_instances = []
    for video_path in video_files:
        # Get the directory of the video
        video_dir = os.path.dirname(video_path)

        # Create a numbered instance folder
        instance_folder = os.path.join(video_dir, f"instance_{instance_counter}_{build_id}")
        os.makedirs(instance_folder, exist_ok=True)

        # Move the video into the instance folder
        video_name = os.path.basename(video_path)
        new_video_path = os.path.join(instance_folder, video_name)
        video_instances.append(new_video_path)
        os.rename(video_path, new_video_path)

        print(f"Moved '{video_path}' to '{new_video_path}'")
        instance_counter += 1
    return video_instances

def save_frame(frame, output_dir, frame_id, video_id) -> str:
    try:
        output_path = os.path.join(output_dir, f'{video_id}_{frame_id}.jpg')
        imageio.imwrite(output_path, frame)
    except Exception as e:
        print("There was an exception: ", e)
    print(f'Frame {frame_id} was succesfully saved')
    return output_path

def read_frame(frame, reader):
    try:
        frame = reader.get_data(frame)
    except Exception as e:
        print('The reader raised an exception:', e)
    return frame

def get_random_frames(num: int, vid: 'video'):
    #vid = video(instance_video_path)
    if num > (vid.frame_count):
        raise ValueError("Cannot select more unique frames than available in range")
    frame_set = sorted(random.sample(range(0, vid.frame_count - 1), num))
    output_dir = os.path.join(vid.dir, 'frames')
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []
    for f in frame_set:
        frame = read_frame(f, vid.reader)
        path = save_frame(frame, output_dir, f, vid.origin_id)
        frame_paths.append(path)
    return frame_paths

def get_specified_frames(frame_set: set[int], vid: 'video'):
    #vid = video(instance_video_path)
    if all(0 < f < vid.frame_count for f in frame_set):
        output_dir = os.path.join(video.dir, 'frames')
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []
        for f in frame_set:
            frame = read_frame(f, vid.reader)
            path = save_frame(frame, output_dir, f, vid.origin_id)
            frame_paths.append(path)
        return frame_paths        
    else:
        raise IndexError(f"Cannot have int in set bigger than {vid.frame_count}, total frame number")

    
# CLASSES

class video:
    '''
    instance_video_path must be the path of a video inside an instance folder already created
    '''
    def __init__(self, instance_video_path: str, origin_id: str):
        print('file exists: ', os.path.isfile(instance_video_path))
        head, tail = os.path.split(instance_video_path)
        self.instance_path = head
        self.video_file_name = tail
        try: 
            probe = ffmpeg.probe(self.video_path)
            print(probe)
        except Exception as e1:
            print("Exception ocurred:", e1)
        fps_info = probe['streams'][0]['r_frame_rate']
        fps = float(eval(fps_info)) 
        self.fps_val = fps
        try:
            frame_count = int(probe['streams'][0]['nb_frames'])
        except:
            print("nb_frames not found for frame_count;\nresorting to duration time and fps...")
            frame_count = int(float(probe['format']['duration'])*fps)
        self.frames = frame_count
        print("Frame count:", frame_count)
        try:
            self.width = int(probe['streams'][0]['width'])
            self.height = int(probe['streams'][0]['height'])
        except Exception as e2:
            print("width and/or height information was not found: ", e2)
        self.origin_id = origin_id
        self.video_reader = imageio.get_reader(instance_video_path, 'ffmpeg')

    def dir(self):
        return self.instance_path
    def path(self):
        return self.video_file_name
    def fps(self):
        return self.fps_val
    def frame_count(self):
        return self.frames
    def size(self):
        return (self.width, self.height)
    def id(self):
         return self.origin_id
    def reader(self):
        return self.video_reader

# MAIN

if __name__ == "__main__":
    # Specify the root directory to start the search
    root_dir = input("Enter the root directory to search for videos: ").strip()
    if os.path.isdir(root_dir):
        video_file_paths = find_files(root_dir)
    else:
        print("The specified directory does not exist.")
    video_file_paths = instance_build(video_file_paths)
    for vidpath in video_file_paths:
        vid = video(vidpath)
        frame_paths = get_random_frames(5, vid) #get_specified_frames can also be used changing this line of code
        df = pd.DataFrame()
        df['frame_path'] = frame_paths
        df.index = list(range(len(frame_paths)))
        # section destined to extracting all descriptions from the frame set obtained



