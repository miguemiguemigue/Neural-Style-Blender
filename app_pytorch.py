from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np

# Check GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)

# image size
img_size = 400

# Load VGG19 pre-trained model, freezing weights
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# feature style layers
STYLE_LAYERS = ['0', '5', '10', '19', '28']
CONTENT_LAYER = ['21']

def get_features(image, model, layers=None):
    """Extract features from layers"""
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def gram_matrix(tensor):
    # Calculate gram matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

@app.route("/nst", methods=["POST"])
def nst_inference():
    # Train step to optimize generated image
    def train_step(generated_image, content_features, style_grams, optimizer, alpha, beta):
        optimizer.zero_grad()

        generated_features = get_features(generated_image, vgg, layers=STYLE_LAYERS + CONTENT_LAYER)

        content_loss = torch.mean((generated_features[CONTENT_LAYER[0]] - content_features[CONTENT_LAYER[0]]) ** 2)

        style_loss = 0
        for layer in STYLE_LAYERS:
            gen_gram = gram_matrix(generated_features[layer])
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((gen_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (generated_features[layer].nelement())

        total_loss = content_loss * alpha + style_loss * beta
        total_loss.backward(retain_graph=True)
        optimizer.step()

        return total_loss

    # Parse alpha and beta from the request, default to 10 and 40 if not provided
    try:
        alpha = float(request.form.get('contentValue', '10'))  # Content weight
        beta = float(request.form.get('styleValue', '40'))    # Style weight
    except:
        # If parsing fails, set to defaults
        alpha = 10
        beta = 40

    # Get images from request
    content_image = request.files['contentImage'].read()
    style_image = request.files['styleImage'].read()

    content_image = preprocess_image(content_image).to(device)
    style_image = preprocess_image(style_image).to(device)


    # Generate initial image. It's the one to optimize in the training steps
    generated_image = content_image.clone().requires_grad_(True)

    # Extract features from content and style images, for the given layers
    content_features = get_features(content_image, vgg, layers=CONTENT_LAYER)
    style_features = get_features(style_image, vgg, layers=STYLE_LAYERS)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

    # Adam optimizer
    optimizer = optim.Adam([generated_image], lr=0.01)

    # Image list to return
    images_to_return_list = []

    # Add initial noisy image
    #images_to_return_list.append(tensor_to_image(generated_image))

    # Train and save intermediate images
    epochs = 300

    num_images_to_return = 3
    if num_images_to_return > 1:
        interval = epochs // (num_images_to_return - 1)
    else:
        interval = epochs  # Only send last image


    interval = epochs // (num_images_to_return - 1) if num_images_to_return > 1 else epochs
    for epoch in range(1, epochs + 1):
        loss = train_step(generated_image, content_features, style_grams, optimizer, alpha, beta)
        print(f'Epoch {epoch}, Loss: {loss.item()}')

        if num_images_to_return > 1:
            # Add start, end and intermediate images
            if epoch == 1 or epoch == epochs or (epoch - 1) % interval == 0:
                images_to_return_list.append(tensor_to_image(generated_image))
        else:
            # Just send last image
            if epoch == epochs:
                images_to_return_list.append(tensor_to_image(generated_image))
    # Convert the generated images to base64
    images_base64 = []
    for img in images_to_return_list:
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
        images_base64.append(img_base64)

    # Return images as JSON
    return jsonify({"images": images_base64})

def preprocess_image(image_data):
    """Preprocess image for Pytorch model."""
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = transform(img)[:3, :, :].unsqueeze(0)
    return img

def tensor_to_image(tensor):
    """Tensor to PIL image."""
    tensor = tensor.cpu().clone().detach()
    tensor = tensor.squeeze(0)
    image = transforms.ToPILImage()(tensor)
    return image

if __name__ == "__main__":
    app.run(debug=True)
