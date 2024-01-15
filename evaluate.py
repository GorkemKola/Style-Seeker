from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
from model.vit import get_model
import os
import random
from PIL import Image
from transformers import ViTImageProcessor
import torch
from annoy import AnnoyIndex
import yaml
from tree.tree import get_similar
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def evaluation(image_tensor, n_recommend):
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

    model = get_model()
    feature_vectors = model(image_tensor)
    feature_vectors = feature_vectors.hidden_states[-1]
    feature_vectors = feature_vectors[:, 0, :]
    print(torch.mean(feature_vectors, axis=1))
    cosine_similarities = []
    correlation_coefficients = []

    for feature_vector in feature_vectors:
        result_paths = get_similar(annoy_index, feature_vector, n_recommend, image_paths)[1:]
        recommended_vectors = np.array([torch.load(os.path.join('data/processed', path)).cpu() for path in result_paths])
        cs = [cosine_similarity([feature_vector], [recommended_vector])[0, 0] for recommended_vector in recommended_vectors]
        coef = [pearsonr(feature_vector, recommended_vector)[0] for recommended_vector in recommended_vectors]
        
        cosine_similarities.append(cs)
        correlation_coefficients.append(coef)

    cs = np.mean(cosine_similarities)
    cs1 = np.mean(cosine_similarities, axis=0)
    cs2 = np.mean(cosine_similarities, axis=1)
    coef = np.array(correlation_coefficients)
    print(cosine_similarities == correlation_coefficients)
    print(cosine_similarities)
    print(correlation_coefficients)
    plt.figure(figsize=(10, 8))
    sns.heatmap(coef, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.xlabel('Recommendation Vectors')
    plt.ylabel('Query Vectors')
    plt.savefig(f'correlation_matrix_heatmap{n_recommend-1}.png')

    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarities, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
            xticklabels=cs1, yticklabels=cs2)
    plt.xlabel('Recommendation Vectors')
    plt.ylabel('Query Vectors')
    plt.title('Cosine Similarity Matrix Heatmap')
    # Save the heatmap as an image file (e.g., PNG)
    plt.savefig(f'cosine_similarity_heatmeap{n_recommend-1}.png')
    return cs
if __name__ == '__main__':
    np.random.seed(42)
    paths = os.listdir('data/processed')
    paths =  np.random.choice(paths, size=20, replace=False)
    processor = ViTImageProcessor('google/vit-base-patch16-224')
    images = []
    for path in paths:
        image = Image.open(os.path.join('data/new', path))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = processor(image)
        images.append(image['pixel_values'][0])

    image_tensor = torch.Tensor(images)
    cs = []
    cs.append(evaluation(image_tensor, 4))
    cs.append(evaluation(image_tensor, 6))
    cs.append(evaluation(image_tensor, 10))
    cs.append(evaluation(image_tensor, 12))
    plt.clf()
    plt.figure(figsize=(4, 2))
    sns.heatmap([cs], annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
            xticklabels=[3, 5, 7, 11])
    
    plt.xlabel('Number of Recommended Images')
    plt.title('Mean Cosine Similarity')
    plt.savefig('mean_cosine_similarity.png')