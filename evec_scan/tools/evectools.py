import moviepy
import ffmpeg
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import random
import ollama

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



    

class ImgDescriptor:
    def __init__(self, model:str, frame_path:str):
        '''
        Uses Ollama.chat models that can take an "image" input with frame_path as its image input variable
        '''
        self.model = model
        self.prompt = "Describe this image"
        self.image_path = frame_path
        self.role = "user"

        res = ollama.chat(model=self.model, messages=[{
            	'role': self.role,
        		'content': self.prompt,
        		'images': [self.image_path]
        }])
        self.response = res
        self.description = res['content']
    
    def get_description(self):
        return self.description
    
    def get_response(self):
        return self.response
    
    def change_prompt(self, new_prompt:str):
        print(f'Old prompt: {self.prompt};\nChanging it to: {new_prompt}')
        self.prompt = new_prompt
        print('Changed the promp successfully')
        

import keras
from keras import applications

class ImgEVecScanner:
    def __init__(self, frame_path:str):
        '''
        Uses Keras.applications models that can take an "image" input with frame_path as its image input variable.
        Since Keras models are imported individually, a list of models is assumed by ImgEvecScanner and can be edited through the classes methods.
        '''
        self.models = [applications.DenseNet121(weights='imagenet', include_top=False),
                       applications.VGG16(weights='imagenet', include_top=False),
                    ]
        self.preprocess_map = {
        "vgg16": applications.vgg16.preprocess_input,
        "vgg19": applications.vgg19.preprocess_input,
        "densenet121": applications.densenet.preprocess_input,
        "densenet169": applications.densenet.preprocess_input,
        }
        self.frame_path = frame_path
        self.models_to_use_indexes = [x for x in range(0,len(self.models))]

    def add_model(self, Keras_applications_model):
        try:
            self.models.append(Keras_applications_model(weights='imagenet', include_top=False))
        except Exception as e:
            return print("Error:", e)
        self.models_to_use_indexes.append(len(self.models)-1)
        return print(f'the new models list is {self.models}\nmodels that will be used have indexes: {self.models_to_use_indexes}')
    
    def get_preprocess_map(self):
        return self.preprocess_map

    def change_preprocess_map(self, new_preprocess_map:dict):
        print(f'Old preprocess_map: {self.preprocess_map}')
        self.preprocess_map = new_preprocess_map
        return print(f'New preprocess_map: {self.preprocess_map}')

    def models_to_use(self, indexes:list):
        print(f'Old indexes of models that would be used: {self.models_to_use_indexes}')
        self.models_to_use_indexes = indexes
        return print(f'New indexes of models that will be used: {self.models_to_use_indexes}')
    
    def get_model_list(self):
        return self.models
    
    def get_models_evecs(self):
        img = keras.utils.load_img(self.frame_path, target_size=(224,224))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims (img_array, axis=0)
        evec_list = []
        names = []
        for index in self.models_to_use_indexes:
            model = self.models[index]
            name = model.name
            names.append(name)
            map = self.preprocess_map.get(name)
            x = map(img_array)
            evec_list.append(model.predict(x))
        return {'model_indexes':self.models_to_use_indexes, 'model_names':names, 'embedding_vectors': evec_list}








class TextEvecScanner:
    def __init__(self, model:str, text:str):
        '''
        Uses Ollama.embeddings models with text as its prompt input variable
        '''
        self.model = model
        self.text = text

        res = ollama.embeddings(
            model=self.model,
            prompt = self.text,)
        
        self.response = res
        self.evec = res['embedding']
    
    def get_response(self):
        return self.response
    
    def get_evec(self):
        return self.evec


# class InstanceGenerator: