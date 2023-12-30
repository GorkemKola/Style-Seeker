import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Load and prepare the ViT model for feature extraction
def get_model(device='cpu'):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_hidden_states=True)
    model = model.to(device)  # Move model to GPU if available

    # Freeze model parameters to avoid gradient calculations during feature extraction
    for param in model.parameters():
        param.requires_grad = False
    
    return model  # Return the modified model

# 3. Load and preprocess the image
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(device)  # Load the modified model
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    image = Image.open('data/images/1163.jpg')  # Load the image
    image_tensor = processor(images=image, return_tensors="pt")
    image_tensor.to(device)
    # 4. Extract image features
    with torch.no_grad():  # Disable gradient calculation for efficiency
        features = model(**image_tensor)
    # 5. Access the extracted vector
    features = features.hidden_states[-1]
    features = torch.sum(features, dim=(1)).squeeze(0)
    print(f'Image vector: {features.shape}')  # Print the vector
