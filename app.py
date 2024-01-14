import yaml
from annoy import AnnoyIndex
import os
from PIL import Image
from transformers import ViTImageProcessor
from model.vit import get_model
import torch
from tree.tree import get_similar
from vit import get_prediction
import gradio as gr


def main(image, slider_value):
    with open('config.yml') as f:
        config = yaml.safe_load(f)

    with open('annoy/image_paths.txt', 'r') as f:
        image_paths = [p[:-1] for p in f.readlines()]

    data_dir = config['data_dir']
    tree_path = config['tree_path']
    metric = config['similarity_metric']
    n_dims = config['n_dims']
    model_name = config['model_name']

    image = Image.fromarray((image * 255).astype('uint8'))

    annoy_index = AnnoyIndex(n_dims, metric=metric)
    annoy_index.load(tree_path)

    processor = ViTImageProcessor(model_name)
    image_tensor = processor(images=image, return_tensors='pt')
    print(image_tensor['pixel_values'].shape)
    model = get_model()
    feature_vector = model(**image_tensor)
    feature_vector = feature_vector.hidden_states[-1]
    feature_vector = feature_vector[0, 0, :]
    print(feature_vector.shape)
    result = get_similar(annoy_index, feature_vector, slider_value, image_paths)
    labels = list(map(lambda x: x[:-4].split('_'), result))
    labels = zip(*labels)
    paths = list(map(lambda path: os.path.join(data_dir, path), result))
    result = []
    images = list(map(lambda x: Image.open(x), paths))
    labels = list(map(lambda x: '\n'.join(x), zip(*labels)))
    labels = '\n=============\n'.join(labels)
    return images, labels


if __name__ == '__main__':
    default_label = "Prediction"
    gallery = gr.Gallery(label="Image Gallery")
    textbox = gr.Textbox(label="Texts")
    iface = gr.Interface(fn=main,
                         inputs=["image",
                                 gr.Slider(minimum=1, maximum=100, step=1,
                                           label="Select how many images will be recommended")],
                         outputs=[gallery, textbox])
    iface.launch()
