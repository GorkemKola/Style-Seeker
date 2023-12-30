from model.vit import get_model
from transformers import ViTImageProcessor
import os
from PIL import Image
import torch
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
data_dir = config['data_dir']
process_dir = config['process_dir']
model_name = config['model_name']

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = get_model('cuda')
    for p in os.listdir(data_dir):
        path = os.path.join(data_dir, p)

        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = processor(images=image, return_tensors='pt')
        image_tensor.to('cuda')
        
        with torch.no_grad():
            features = model(**image_tensor)

        
        # 5. Access the extracted vector
        features = features.hidden_states[-1]
        features = torch.sum(features, dim=(1)).squeeze(0)
        out_path = os.path.join(process_dir, p)

        torch.save(features, out_path)