from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
# Allow CORS temporarily
CORS(app)

# image size
img_size = 400

# Load VGG19 pre-trained model
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='../model/vgg_pretrained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False

#tf.config.run_functions_eagerly(True)

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]



@app.route("/nst", methods=["POST"])
def nst_inference():

    # train step function to optimize generated image (get close to Content, but applying style)
    @tf.function
    def train_step(generated_image):
        with tf.GradientTape() as tape:
            a_G = vgg_model_outputs(generated_image)
            J_style = compute_style_cost(a_S, a_G)
            J_content = compute_content_cost(a_C, a_G)
            J = total_cost(J_content, J_style)
        grad = tape.gradient(J, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
        return J


    # Get images from request
    content_image = request.files['contentImage'].read()
    style_image = request.files['styleImage'].read()

    content_image = bytes_to_resized_ndarray(content_image, img_size)
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

    style_image = bytes_to_resized_ndarray(style_image, img_size)
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    generated_image = tf.Variable(generated_image)

    content_layer = [('block5_conv4', 1)]
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs(style_image)  # Style encoder

    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    a_G = vgg_model_outputs(generated_image)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # List of images to return
    images_to_return_list = []

    # Guardar la imagen inicial (ruido)
    images_to_return_list.append(tensor_to_image(generated_image))

    # Train for the specified number of epochs and save intermediate images
    epochs = 100
    checkpoints = [1, 25, 50, 75, 100]
    for i in range(1, epochs + 1):
        train_step(generated_image)
        print('Epoch {}...'.format(i))

        # save images at checkpoints
        if i in checkpoints:
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

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.reshape(tf.transpose(a_S), shape=[n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G), shape=[n_C, n_H * n_W])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * (n_C ** 2) * ((n_H * n_W) ** 2))


    return J_style_layer


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    #
    GA = tf.matmul(A, tf.transpose(A))


    return GA


def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]


    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape 'a_C' and 'a_G'
    a_C_unrolled = tf.reshape(a_C, shape=[n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[n_H * n_W, n_C])

    # compute the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content


@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    #
    J = alpha * J_content + beta * J_style


    return J


def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1

    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Arguments:
    tensor -- Tensor

    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def preprocess_image(image_data):
    # Convert image data to tensor for VGG19
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    return tf.convert_to_tensor([img_array], dtype=tf.float32)


def vgg_model_outputs(image_tensor):
    """
    Get the outputs from the VGG19 model to get the relevant layer outputs for style and content
    :param image_tensor:
    :return:
    """
    outputs = vgg(image_tensor)
    return outputs


def bytes_to_resized_ndarray(image_bytes, img_size):
    """
    Convert image bytes to a resized ndarray image
    :param image_bytes:
    :param img_size:
    :return:
    """
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image_resized = image.resize((img_size, img_size), Image.LANCZOS)  # Puedes usar otros métodos de redimensionamiento
    image_ndarray = np.array(image_resized)

    return image_ndarray

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


if __name__ == "__main__":
    app.run(debug=True)
