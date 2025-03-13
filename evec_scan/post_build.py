# IMPORTS

from sklearn.neighbors import NearestNeighbors
import glob
import numpy as np
import pandas as pd
import time
import os

# FUNCTIONS

def find_directories(path):
    """Find all directories inside the given path."""
    return [d for d in glob.glob(os.path.join(path, "*/")) if os.path.isdir(d)]

def find_numpy_file(path):
    """Find all .npy files inside the given directory."""
    file = glob.glob(os.path.join(path, "*.npy"))
    return file

def euclidean_distance(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    
    return np.linalg.norm(v1 - v2)

def jaccard_similarity(s1, s2):
    s1, s2 = set(s1), set(s2)  # Convert to sets for easy comparison
    intersection = len(s1 & s2)  # Number of common indexes
    union = len(s1 | s2)  # Total unique indexes
    return intersection / union if union != 0 else 0.0  # Avoid division by zero

def cluster_similar(evec_group, k):
    evec_group = np.array(evec_group)  # Ensure input is a NumPy array
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')  # k+1 to exclude itself
    knn.fit(evec_group)
    distances, indices = knn.kneighbors(evec_group)
    
    # Exclude self index
    indices_list = [(i, indices[i][1:].tolist()) for i in range(len(evec_group))]
    
    return indices_list

# CLASSES

class instance:
    def __init__(self, dir: str):
        self.dir = dir
        self.name, self.root = os.path.split(dir)
        dpath, fpath, ipath  = os.path.join(dir,'desc_evecs'),  os.path.join(dir,'frames'), os.path.join(dir,'img_evecs')
        if os.path.isdir(dpath) and os.path.isdir(fpath) and os.path.isdir(ipath): 
            self.desc_evecs_dir, self.frames_dir, self.img_evecs_dir = dpath, fpath, ipath
            print("Valid instance directory...")
        else: 
            raise Exception(f"{dir} is not a valid instance directory")
        self.dataframe = pd.read_csv(os.path.join(dir,'instance_data.csv'))
    def get_dir(self):
        return self.dir
    def get_name(self):
        return self.name
    def get_root(self):
        return self.root
    def get_desc_evecs_dir(self):
        return self.desc_evecs_dir
    def get_frames_dir(self):
        return self.frames_dir
    def get_img_evecs_dir(self):
        return self.img_evecs_dir
    def get_df(self):
        return self.dataframe
    

# MAIN SCRIPT

if __name__ == "__main__":

    start_build = time.time()
    # parameters and global especifications
    K = 3

    root_dir = input("Enter the instance directory: ").strip()
    if not os.path.isdir(root_dir):
        raise  Exception(f"{root_dir} is not a valid instance directory")
    else:
        pass
    inst = instance(root_dir)
    
    # descriptions
    desc_dir_arrays = []
    dirs = find_directories(inst.get_desc_evecs_dir())
    for dir in dirs:
        for file in find_numpy_file(dir):
            desc_dir_arrays.append((dir, file ,cluster_similar(np.load(file), K)))
    
    # images
    img_dir_arrays = []
    dir = inst.get_img_evecs_dir()
    for file in find_numpy_file(dir):
        desc_dir_arrays.append((dir, file ,cluster_similar(np.load(file), K)))

    # comparison one by one
    comparison_df = pd.DataFrame(columns=['id','similar_desc_ids', 'similar_img_ids', 'jaccard_similarity'])
    

    


    


