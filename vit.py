import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load the ViT model for feature extraction
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_hidden_states=True)
model = model.to('cpu')  # Move model to CPU for Gradio

# Freeze model parameters to avoid gradient calculations during feature extraction
for param in model.parameters():
    param.requires_grad = False

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to model input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])


# Define a function for model inference
def get_prediction(image, slider_value):
    # Preprocess the image
    image = Image.fromarray((image * 255).astype('uint8'))  # Convert Gradio image to PIL
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Get the prediction
    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = model(image)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return str(predicted_class)  # Return prediction as a string


# Create Gradio Interface


