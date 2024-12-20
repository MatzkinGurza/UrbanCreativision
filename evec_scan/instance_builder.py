import numpy as np
import os
from keras import applications
from tools import evectools as evt
import pandas as pd
from keras import applications
######################################################################################################
df = pd.DataFrame()
instance_dir = 'evec_scan/instances/test1'
vid_file = 'vid3.mp4'
instance_id = 'test1'
group_size = 2
vidpath = 'evec_scan/instances/test1/vid3.mp4'
description_models = ['moondream'] #'moondream', 'llava-llama3'
description_embedding_models = {"model":['mxbai-embed-large', "ViT-B/32"], "model_origin":['ollama', "clip"]} #'mxbai-embed-large','nomic-embed-text', 'all-minilm' from ollama
image_embedding_models = {'model':[applications.VGG16, "ViT-B/32"], 'model_origin':['keras', 'clip']} #applications.VGG16, applications.DenseNet121
#######################################################################################################
video_instance = evt.FrameExtractor(vidpath=vidpath)
frame_group = video_instance.get_random_frame_group(size=group_size)
frame_group_instance = video_instance.FrameGroup(frame_extractor=video_instance, group_set=frame_group)
frames_output_path = os.path.join(instance_dir, 'frames')
frame_path_list = frame_group_instance.save_group(output_dir=frames_output_path, group_name=instance_id)
df['frame_path'] = frame_path_list
df.index = list(range(len(frame_path_list)))
########################################################################################################
for desc_model in description_models: 
    descriptions_per_model = []
    description_columns = []
    for frame_path in df['frame_path']:
        descriptor = evt.ImgDescriptor(model=desc_model,frame_path=frame_path)
        description = descriptor.get_description()
        descriptions_per_model.append(description)
    df[f'{desc_model}_desc'] = descriptions_per_model
    description_columns.append(f'{desc_model}_desc')
df_output_path = os.path.join(instance_dir, 'instance_data.csv')
df.to_csv(df_output_path, index=True)
#######################################################################################################
for dir in description_columns:
    description_evec_output_dir = os.path.join(instance_dir, 'desc_evecs', dir)
    os.makedirs(description_evec_output_dir)
    for text_evec_model, model_origin in zip(description_embedding_models['model'],description_embedding_models['model_origin']):
        description_evec_arr_per_model_per_desc_model = np.empty(group_size, dtype=object)
        i = 0
        for description in df[dir]:
            text_evec_scanner = evt.TextEvecScanner(model=text_evec_model, text=description, model_origin=model_origin)
            text_evec = text_evec_scanner.get_evec()
            description_evec_arr_per_model_per_desc_model[i]=text_evec
            i += 1
        output_path_desc_evec = os.path.join(description_evec_output_dir,f'{text_evec_model}.npy')
        np.save(file=output_path_desc_evec, arr=description_evec_arr_per_model_per_desc_model)
#######################################################################################################
    img_evecs_output_dir = os.path.join(instance_dir, 'img_evecs')
    os.makedirs(img_evecs_output_dir)
    for image_evec_model, model_origin in zip(image_embedding_models['model'],image_embedding_models['model_origin']):
        image_evec_arr_per_model_per_img_model = np.empty(group_size, dtype=object)
        img_evec_scanner = evt.ImgEVecScanner()
        if model_origin.lower() == "keras":
            img_evec_scanner.add_model(Keras_applications_model=image_evec_model)
        for frame_path in df['frame_path']:
            i = 0
            if model_origin.lower() == "keras":
                res_dict = img_evec_scanner.get_models_evecs(frame_path=frame_path, lin_method='GAP')
            elif model_origin.lower() == "clip":
                res_dict = img_evec_scanner.get_clip_evec(model=image_evec_model, frame_path=frame_path)
            for name, evec in zip(res_dict['model_name'],res_dict['embedding_vector']):
                output_path_img_evec = os.path.join(img_evecs_output_dir,f'{name}.npy')
                image_evec_arr_per_model_per_img_model[i] = evec[0]
                i += 1
        np.save(file=output_path_img_evec, arr=image_evec_arr_per_model_per_img_model)

        





# #### DESCOBRIR COMO SALVAR OS EMBEDDINGS E ORGANIZAR O JSON RESULTANTE

# InstanceGen = evt.InstanceGenerator(instance_dir='evec_scan/instances/test1', vid_file='vid3.mp4', instance_name='test_instance3')
# print('Instance created\nStructure: ', InstanceGen.get_structure())
# InstanceGen.generate_instance_frames(frame_quant=3)
# print('Instance frames generated\nStructure: ', InstanceGen.get_structure())
# InstanceGen.get_descriptions(ollama_model_list=['moondream', 'llava-llama3'])
# print('Instance frames described\nStructure: ', InstanceGen.get_structure())
# InstanceGen.get_description_embeddings(ollama_model_list=['mxbai-embed-large','nomic-embed-text', 'all-minilm'])
# print('Instance descriptions embedded\nStructure: ', InstanceGen.get_structure())
# structure = InstanceGen.get_structure()
# structure['image_embedding_models'] = []
# structure['image_embedding_vectors'] = []
# structure['image_embedding_vectors_shape'] = []


# for frame_path in structure['frame_paths']:
#     frame_evec = evt.ImgEVecScanner(frame_path=frame_path)
#     # frame_evec.add_model(Keras_applications_model=applications.VGG19)
#     # frame_evec.change_preprocess_map(...)
#     # frame_evec.change_models_to_use(...)
#     # frame_evec.get_model_list(...)
#     # frame_evec.get_preprocess_map(...)
#     ImgEvecDict = frame_evec.get_models_evecs(lin_method='GAP')
#     print('image embedding data', ImgEvecDict)
#     structure['image_embedding_models'].append(ImgEvecDict['model_name'])
#     structure['image_embedding_vectors'].append(ImgEvecDict['embedding_vector'])
#     structure['image_embedding_vectors_shape'].append(ImgEvecDict['vector_shape'])

# print(structure)


