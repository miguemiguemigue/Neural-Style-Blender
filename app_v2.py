from flask import Flask, request, jsonify
from utils.config_loader import load_config

app = Flask(__name__)

# Load config mode
mode = load_config()

if mode == 'tensorflow':
    from models.tensorflow.tensorflow_model import load_vgg_model, run_style_transfer
elif mode == 'pytorch':
    from models.pytorch.pytorch_model import load_vgg_model, run_style_transfer
else:
    raise ValueError(f"Unknown mode: {mode}. Must be 'tensorflow' or 'pytorch'.")

# Load vgg model
vgg_model = load_vgg_model()

@app.route('/nst', methods=['POST'])
def nst_inference():
    content_image = request.files['contentImage'].read()
    style_image = request.files['styleImage'].read()

    # Preprocesar imágenes (se puede usar una función común)
    content_image = preprocess_image(content_image, img_size=400)
    style_image = preprocess_image(style_image, img_size=400)

    # Ejecutar la transferencia de estilo según el framework cargado
    generated_image = run_style_transfer(content_image, style_image)

    # Convertir el tensor generado a imagen
    result_image = tensor_to_image(generated_image)

    # Devolver la imagen generada en base64 para la respuesta JSON
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

    return jsonify({"image": img_base64})

if __name__ == '__main__':
    app.run(debug=True)
