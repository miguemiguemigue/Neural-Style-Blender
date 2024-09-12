from PIL import Image
import numpy as np
import io

def preprocess_image(image, img_size):
    # preprocess image
    image = Image.open(io.BytesIO(image)).resize((img_size, img_size))
    image = np.array(image) / 255.0
    return image

def tensor_to_image(tensor):
    # convert tensor to image
    image = tensor.squeeze()  # remove batch dimension if needed
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype(np.uint8))