import moviepy
import ffmpeg
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import random


class FrameExtractor:
    def __init__(self, vidpath:str):

        self.video_path = vidpath 

        try: 
            probe = ffmpeg.probe(self.video_path)
        except Exception as e:
            print("Error:", e)
        # print(probe)

        fps_info = probe['streams'][0]['r_frame_rate']
        fps = float(eval(fps_info)) 

        try:
            frame_count = int(probe['streams'][0]['nb_frames'])
        except:
            print("nb_frames not found for frame_count;\nresorting to duration time and fps...")
            frame_count = int(float(probe['format']['duration'])*fps)
        # print(frame_count)

        width = int(probe['streams'][0]['width'])

        height = int(probe['streams'][0]['height'])

        self.fps = fps
        self.frame_count = frame_count
        self.reader = imageio.get_reader(self.video_path, 'ffmpeg')
        self.width = width
        self.height = height

    def get_frames_from_list(self, frame_set:set[int]) -> dict:
        if len(frame_set)>self.frame_count:
            return print(f"Too many frames on set\nframe_set: {frame_set}\ntotal frames: {self.frame_count}")
        else:
            try:
                frame_list = sorted(list(frame_set))
                frames = []
                for frame in frame_list:
                    if frame >= self.frame_count:
                        print(f"Warning: Frame {frame} is out of bounds. Skipping...")
                        continue 
                    else: 
                        frames.append(self.reader.get_data(frame))                          

            except Exception as e:
                return print("Error:", e)
                
            return {'frame_number': frame_list, 'frame': frames}
   
    def get_frame_count(self):
        """
        Returns the total number of frames in the video.
        """
        return self.frame_count

    def get_frame_rate(self):
        """
        Returns the frame rate of the video.
        """
        return self.fps
    
    def get_random_frame_group(self, size:int):
        """
        returns a random set of frames.
        """
        return set([random.randint(0, self.frame_count-1) for _ in range(size)])

    class FrameGroup:
        def __init__(self, frame_extractor: 'FrameExtractor', group_set: set[int]):
            self.frame_extractor = frame_extractor
            self.group = group_set
            self.structure = frame_extractor.get_frames_from_list(group_set)
            self.frames = self.structure['frame']
            self.groupname = ''
        
        def show_frame(self, frame_num: int):
            frame_index = self.structure['frame_number'].index(frame_num) 
            frame = self.structure['frame'][frame_index]
            # Display the frame using matplotlib
            plt.imshow(frame)
            plt.axis('off')  # Hide axes
            plt.show()

        def save_group(self, group_name:str, output_dir:str) -> str:
            # Ensure the output directory exists
            self.groupname = group_name
            os.makedirs(output_dir, exist_ok=True)
            try:
                for num, frame in zip(*self.structure.values()):
                    output_path = os.path.join(output_dir, f'{group_name}_{num}.jpg')
                    imageio.imwrite(output_path, frame)
            except Exception as e:
                return print("Error:", e)
            return print('files were succesfully saved')



    

# class ImgDescriptor():


# class ImgEVecScanner():


# class TextEvecScanner():


# class InstanceGenerator():