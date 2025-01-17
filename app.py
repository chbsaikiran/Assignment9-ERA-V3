import torch
import torchvision.transforms as transforms
from torchvision import models
import gradio as gr
from PIL import Image

# Load the ResNet50 model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 1000)  # Update for 1000 classes

# Load model state dict from checkpoint
checkpoint = torch.load("model_32.pth", map_location="cpu")
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# ImageNet class labels
imagenet_classes = [line.strip().split(maxsplit=1)[1] for line in open("imagenet_classes.txt")]

# Define image preprocessing
def preprocess_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


# Define prediction function
def predict(image):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = []
    for i in range(top5_prob.size(0)):
        results.append((imagenet_classes[top5_catid[i]], top5_prob[i].item()))
    return results

# Gradio Interface
def classify_image(image):
    if image is None:
        return {"Error": "No image uploaded. Please upload a valid image."}
    try:
        results = predict(image)
        # Convert results into the correct format
        return {r[0]: r[1] for r in results}
    except Exception as e:
        return {"Error": f"Prediction failed: {e}"}



interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet50 Image Classifier",
    description="Upload an image to get the top 5 predictions using a ResNet50 model trained on ImageNet."
)

if __name__ == "__main__":
    interface.launch()
