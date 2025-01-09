import moviepy
import ffmpeg
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import random
import ollama
import torch
import clip
from PIL import Image

class FrameExtractor:
    def __init__(self, vidpath:str):

        self.video_path = vidpath 
        print('file exists: ', os.path.isfile(vidpath))

        try: 
            probe = ffmpeg.probe(self.video_path)
            print(probe)

        except Exception as e:
            print("Error:", e)
        

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

        def get_structure(self):
            return self.structure

        def show_frame(self, frame_num: int):
            frame_index = self.structure['frame_number'].index(frame_num) 
            frame = self.structure['frame'][frame_index]
            # Display the frame using matplotlib
            plt.imshow(frame)
            plt.axis('off')  # Hide axes
            plt.show()

        def save_group(self, group_name:str, output_dir:str) -> list:
            # Ensure the output directory exists
            self.groupname = group_name
            os.makedirs(output_dir, exist_ok=True)
            try:
                frame_paths = []
                for num, frame in zip(*self.structure.values()):
                    output_path = os.path.join(output_dir, f'{group_name}_{num}.jpg')
                    frame_paths.append(output_path)
                    imageio.imwrite(output_path, frame)
            except Exception as e:
                return print("Error:", e)
            print('files were succesfully saved')
            return frame_paths



    

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
        self.description = res['message']['content']
    
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
from typing import Literal, Optional
from keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D

class ImgEVecScanner:
    def __init__(self):
        '''
        Uses Keras.applications models that can take an image input with frame_path as its image input variable.<p> 
        Since Keras models are imported individually, a list of models is assumed by ImgEvecScanner and can be edited through the classes methods.<p>
        Obs) As a new addition, a method to extract the embedding through OpenAIs clip models is available 
        '''
        self.models =[]
        # self.models = [applications.DenseNet121(weights='imagenet', include_top=False),
        #                applications.VGG16(weights='imagenet', include_top=False),
        #             ]
        self.preprocess_map = {
        "vgg16": applications.vgg16.preprocess_input,
        "vgg19": applications.vgg19.preprocess_input,
        "densenet121": applications.densenet.preprocess_input,
        "densenet169": applications.densenet.preprocess_input,
        }
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

    def change_models_to_use(self, indexes:list):
        print(f'Old indexes of models that would be used: {self.models_to_use_indexes}')
        self.models_to_use_indexes = indexes
        return print(f'New indexes of models that will be used: {self.models_to_use_indexes}')
    
    def get_model_list(self):
        return self.models
    
    def get_models_evecs(self, frame_path:str, lin_method:Literal['Flatten', 'GMP', 'GAP' ]):
        img = keras.utils.load_img(frame_path, target_size=(224,224))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims (img_array, axis=0)
        evec_list = []
        names = []
        vec_dtypes =[]
        vec_shapes = []
        for index in self.models_to_use_indexes:
            model = self.models[index]
            name = model.name
            names.append(name)
            map = self.preprocess_map.get(name)
            x = map(img_array)
            if lin_method=='GMP':
                lin_layer=GlobalMaxPooling2D()
            elif lin_method=='GAP':
                lin_layer=GlobalAveragePooling2D()
            elif lin_method=='Flatten':
                lin_layer=Flatten()
            tensor = lin_layer(model.predict(x))
            evec_list.append(tensor.numpy())
            vec_dtypes.append(tensor.dtype)
            vec_shapes.append(tensor.shape)
        return {'model_index':self.models_to_use_indexes, 'model_name':names, 
                'embedding_vector': evec_list, 'vector_shape':vec_shapes, 'vector_dtype':vec_dtypes}

    def get_clip_evec(self, model, frame_path):
        image = Image.open(frame_path).convert("RGB")
        image = image.resize((224, 224))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, preprocess = clip.load(model, device=device)
        processed_image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clipmodel.encode_image(processed_image)
        return {"embedding_vector":[image_embedding.cpu().numpy()],"model_name":[f"clip_{str(model).replace("/","")}"]}


class TextEvecScanner:
    def __init__(self, model:str, text:str, model_origin:str):
        '''
        Uses Ollama.embeddings models with text as its prompt input variable
        or uses the clip from openai if flag is set to clip
        '''
        self.model = model
        self.text = text

        if model_origin.lower() =='clip':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(model, device=device)
            text_tokens = clip.tokenize(text).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokens)
            res = text_embedding

        elif model_origin.lower() == 'ollama':
            res = ollama.embeddings(
            model=self.model,
            prompt = self.text,)
        
        self.response = res
        self.evec = np.array(res['embedding'])
    
    def get_response(self):
        return self.response
    
    def get_evec(self):
        return self.evec
    
    def get_clip_available_models():
        return clip.available_models()
    


class InstanceGenerator:
    def __init__(self, instance_dir:str, vid_file:str, instance_name:str):
        '''
        to build an instance, you must create a directory containing the video file to be anayzed in the instance.<p> 
        It is recommended to have only one video file per instance directory for organization purposes.<p> 
        The instance_name attribute will be used as a part of resulting instance paths, considering that, do not use big names, neither blank spaces or punctuation.<p> 
        vid_path must be the name of the video file inside the insance_dir, including extension like ".mp4".<p> 
        InstanceGenerator is a way to facilitate the process of creating an instance of model descriptions and its embedding vectors for a series of frames. 
        Due to ImgEvecScanner already having methods tha facilitate applying many models to a single frame, its usage has not been implemented as a method inside this class.
        '''
        self.instance_path = instance_dir
        self.vidpath = os.path.join(instance_dir, vid_file)
        self.frame_group_name = instance_name
        self.structure = None
    
    def generate_instance_frames(self, 
                                 frame_quant:Optional[int], 
                                 frame_num_list:Optional[set[int]]=None, 
                                 from_list:bool=False):
        '''
        if from_list set to True, frame_num_list is necessary and should receive a set of the frame numbers to be extracted.<p> 
        if from_list set to False, frame_quant is necessary and should receive the number of frames to be extracted.<p> 
        '''
        extractor = FrameExtractor(self.vidpath)
        if from_list:
            fgp = extractor.FrameGroup(extractor, frame_num_list)
        else:
            rfgp = extractor.get_random_frame_group(size=frame_quant)
            fgp = extractor.FrameGroup(extractor, rfgp)
            output_dir = os.path.join(self.instance_path, 'frames')
            frame_paths = fgp.save_group(group_name=self.frame_group_name, output_dir=output_dir)
            structure = fgp.get_structure()
            structure['frame_paths'] = frame_paths
            self.structure = structure
        return structure #this is a dictionary with the image arrays, the frame numbers and the saved frame paths
    
    def get_descriptions(self, ollama_model_list:str):
        structure = self.structure
        if not structure:
            return print('before getting decriptin you must get instance_frames')
        else:
            structure['descriptions'] = []
            for frame_path in structure['frame_paths']:
                description_list = []
                for model in ollama_model_list: 
                    descriptor = ImgDescriptor(model, frame_path=frame_path)
                    description_list.append(descriptor.get_description())
                structure['descriptions'].append(description_list)
            structure['description_models'] = ollama_model_list
            self.structure = structure
        return structure #this is a dictionary with the image arrays, the frame numbers and the saved frame paths, descriptions and description models
    
    def get_description_embeddings(self, ollama_model_list:list[str]):
        structure = self.structure
        if not structure['descriptions']:
            return print('before getting embedding vector of description you must get_descriptions')
        else:
            structure['description_evecs'] = []
            for frame_descriptions in structure['descriptions']:
                desc_evec_list = []
                for desc in frame_descriptions:
                    model_evec_list = []
                    for model in ollama_model_list: 
                        txtevec = TextEvecScanner(model, text=desc)
                        model_evec_list.append(txtevec.get_evec())
                    desc_evec_list.append(model_evec_list)
                structure['description_evecs'].append(desc_evec_list)
            structure['text_embedding_models'] = ollama_model_list
            self.structure = structure
        return structure #this is a dictionary with the image arrays, the frame numbers and the saved frame paths, descriptions and description models
    
    def get_structure(self):
        return self.structure