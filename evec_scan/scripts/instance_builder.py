import ffmpeg
import numpy as np
import imageio
import os
import random
import ollama
import keras
from keras import applications
from evec_scan.tools import evectools as evt
import pandas as pd

#### DESCOBRIR COMO SALVAR OS EMBEDDINGS E ORGANIZAR O JSON RESULTANTE

InstanceGen = evt.InstanceGenerator(instance_dir='evec_scan/instances/test1', vid_file='vid3.mp4', instance_name='test_instance3')
print('Instance created\nStructure: ', InstanceGen.get_structure())
InstanceGen.generate_instance_frames(frame_quant=3)
print('Instance frames generated\nStructure: ', InstanceGen.get_structure())
InstanceGen.get_descriptions(ollama_model_list=['moondream', 'llava-llama3'])
print('Instance frames described\nStructure: ', InstanceGen.get_structure())
InstanceGen.get_description_embeddings(ollama_model_list=['mxbai-embed-large','nomic-embed-text', 'all-minilm'])
print('Instance descriptions embedded\nStructure: ', InstanceGen.get_structure())
structure = InstanceGen.get_structure()
structure['image_embedding_models'] = []
structure['image_embedding_vectors'] = []
structure['image_embedding_vectors_shape'] = []


for frame_path in structure['frame_paths']:
    frame_evec = evt.ImgEVecScanner(frame_path=frame_path)
    # frame_evec.add_model(Keras_applications_model=applications.VGG19)
    # frame_evec.change_preprocess_map(...)
    # frame_evec.change_models_to_use(...)
    # frame_evec.get_model_list(...)
    # frame_evec.get_preprocess_map(...)
    ImgEvecDict = frame_evec.get_models_evecs(lin_method='GAP')
    print('image embedding data', ImgEvecDict)
    structure['image_embedding_models'].append(ImgEvecDict['model_name'])
    structure['image_embedding_vectors'].append(ImgEvecDict['embedding_vector'])
    structure['image_embedding_vectors_shape'].append(ImgEvecDict['vector_shape'])

print(structure)
