from utils import *

# fill your path from your local folder path here
root_path = './MICCAI_2019_pathology_challenge/'
path_to_save_labels = './labels'
make_dirs(path_to_save_labels)

# Read paths from 6 annotators
maps = read_labels(root_path)
img_paths = sorted(glob.glob(os.path.join(root_path,'Train Imgs/Train Imgs/*')))
print('Imgs found:', len(img_paths))
assert len(img_paths)==244 , 'Check your path'
# Majority voting and save image label
preprocess_labels(maps, img_paths, path_to_save_labels)



