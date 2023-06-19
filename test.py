import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Load the pre-trained ResNet18 model
model = resnet18(pretrained=True)
model.eval()

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image and preprocess it
image_path = "appy.png"
image = Image.open(image_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Check if a GPU is available and move the model to the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_batch = input_batch.to(device)

# Perform the inference
with torch.no_grad():
    input_batch = input_batch.to(device)
    output = model(input_batch)

# Apply softmax to get the confidence scores
probs = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted label
_, predicted_label = torch.max(probs, 0)

# Print the confidence value and predicted label
confidence = probs[predicted_label.item()].item()
predicted_class = "Apple" if predicted_label.item() == 0 else "Not Apple"
print(f"Confidence: {confidence:.4f}")
print(f"Predicted Class: {predicted_class}")
