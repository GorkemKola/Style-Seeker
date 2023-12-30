import yaml
from annoy import AnnoyIndex
import os
from PIL import Image
from transformers import ViTImageProcessor
from model.vit import get_model
import torch
from tree.tree import get_similar

def main():
    with open('config.yml') as f:
        config = yaml.safe_load(f)

    with open('annoy/image_paths.txt', 'r') as f:
        image_paths = [p[:-1] for p in f.readlines()]

    data_dir = config['data_dir']
    tree_path = config['tree_path']
    metric = config['similarity_metric']
    n_dims = config['n_dims']
    model_name = config['model_name']
    annoy_index = AnnoyIndex(n_dims, metric=metric)
    annoy_index.load(tree_path)
    path = os.path.join(data_dir, image_paths[0])
    image = Image.open('/home/grkmkola/Desktop/Projects/StyleSeeker/data/images/1525.jpg')
    processor = ViTImageProcessor(model_name)
    image_tensor = processor(images=image, return_tensors='pt')
    model = get_model()
    feature_vector = model(**image_tensor)
    feature_vector = feature_vector.hidden_states[-1]
    feature_vector = torch.sum(feature_vector, dim=(1)).squeeze(0)
    print(get_similar(annoy_index, feature_vector, 5, image_paths))

if __name__ == '__main__':
    main()
