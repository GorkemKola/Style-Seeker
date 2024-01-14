from transformers import ViTImageProcessor
from PIL import Image
from annoy import AnnoyIndex
import torch
import os
import yaml


def construct():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    process_dir = config['process_dir']
    model_name = config['model_name']
    n_dims = config['n_dims']
    n_trees = config['n_trees']
    annoy_dir = config['annoy_dir']
    metric = config['similarity_metric']
    os.makedirs(annoy_dir, exist_ok=True)

    feature_paths = os.listdir(process_dir)

    annoy_index = AnnoyIndex(n_dims, metric=metric)
    paths = []
    for i, p in enumerate(feature_paths):
        path = os.path.join(process_dir, p)
        features = torch.load(path)
        print(p, i)
        annoy_index.add_item(i, features)
        paths.append(p)

    annoy_index.build(n_trees)

    annoy_index.save(os.path.join(annoy_dir, 'annoy_idx.ann'))

    with open(os.path.join(annoy_dir, 'image_paths.txt'), 'w') as f:
        for image_path in paths:
            f.write(f'{image_path}\n')

    print('ANNOY tree saved.')


def get_similar(tree, query_vector, num_neighbors, image_paths):
    similar_indices = tree.get_nns_by_vector(query_vector, num_neighbors)
    similar_image_paths = [image_paths[index] for index in similar_indices]
    return similar_image_paths


if __name__ == "__main__":
    construct()
