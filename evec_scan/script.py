# IMPORTS

import os
import glob
import random
from typing import Literal, Optional
import ffmpeg
import imageio
import pandas as pd
import ollama
import torch
import clip
import numpy as np

        

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
        self.vidpath = instance_video_path
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
        return self.vidpath
    def video(self):
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
    
class descriptor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session = ollama.ChatSession(model=model_name)

    def describe(self, frame_path):
        with open(frame_path, "rb") as frame_file:
            image_bytes = frame_file.read()

        response = self.session.send_message({
            "role": "user",
            "content": "Describe this image in detail.",
            "images": [image_bytes]  # Send the image as input
        })
        return response["message"]["content"]

class text_embedder:
    def __init__(self, model, origin):
        """
        Keeps the model loaded and processes multiple texts efficiently.
        Supports either Ollama embeddings or OpenAI CLIP embeddings.
        """
        self.model_origin = origin.lower()
        self.model_name = model
        
        if self.model_origin == 'clip':
            # Load CLIP model only once
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model, device=self.device)
        
        elif self.model_origin == 'ollama':
            # Keep Ollama model loaded in memory
            self.session = ollama.EmbeddingSession(model=model)
        
    def compute_embedding(self, text: str):
        """
        Compute the embedding for a given text.
        Uses the preloaded model to avoid redundant initializations.
        """
        if self.model_origin == 'clip':
            text_tokens = clip.tokenize(text).to(self.device)
            with torch.no_grad():
                text_embedding = self.model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()
        
        elif self.model_origin == 'ollama':
            res = self.session.get_embedding(prompt=text)  # Uses preloaded Ollama session
            return np.array(res['embedding'])
        

# MAIN

if __name__ == "__main__":

    # Global especifications
    NUM_FRAMES = 5 # number of frames to be extracted at random per video
    DESC_MODELS = ['moondream', 'llava-llama3']
    DESC_EMBEDDERS = {"model":['mxbai-embed-large'], "model_origin":['ollama']}

    # Specify the root directory to start the search
    root_dir = input("Enter the root directory to search for videos: ").strip()
    if os.path.isdir(root_dir):
        video_file_paths = find_files(root_dir)
    else:
        print("The specified directory does not exist.")
    video_file_paths = instance_build(video_file_paths)

    # this section runs the program for each instance found in the previous directory search
    for vidpath in video_file_paths:
        vid = video(vidpath)
        frame_paths = get_random_frames(NUM_FRAMES, vid) #get_specified_frames can also be used changing this line of code but would demand change in NUM_FRAMES to match
        df = pd.DataFrame()
        df['frame_path'] = frame_paths
        df.index = list(range(len(frame_paths)))
        
        # section destined to extracting all descriptions from the frame set obtained
        for desc_model in DESC_MODELS: 
            model = descriptor(desc_model)
            descriptions_per_model = []
            for frame_path in frame_paths:
                description = model.describe(frame_path=frame_path)
                descriptions_per_model.append(description)
            df[f'{desc_model}'] = descriptions_per_model
            os.makedirs(os.path.join(vid.dir, 'desc_evecs', f'{desc_model}')) # creates a directory for the embeddings for each descriptor

        # saves the dataframe to te instance directory
        df_output_path = os.path.join(vid.dir, 'instance_data.csv')
        df.to_csv(df_output_path, index=True)     

        # This section gets the embedding vector of each frames description, 
        # desc_model by desc_model for all description embedding models
        for model, origin in zip(DESC_EMBEDDERS['model'],DESC_EMBEDDERS['model_origin']):
            embedder = text_embedder(model, origin)
            for desc_model in DESC_MODELS:
                evec_array = np.empty(NUM_FRAMES, dtype=object)
                i = 0
                for desc in df[desc_model]:
                    evec = embedder.compute_embedding(desc)
                    evec_array[i] = evec
                    i += 1
                output_desc_evec = os.path.join(vid.dir, 'desc_evecs', f'{desc_model}', f'{str(model).replace('/','')}.npy')
                np.save(file=output_desc_evec, arr=evec_array)
        
        # This section gets the embedding vector for each frame model by model

