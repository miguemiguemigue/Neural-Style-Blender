import tensorflow as tf



def load_vgg_model(img_size = 400):
    vgg = tf.keras.applications.VGG19(include_top=False, input_shape=(img_size, img_size, 3))
    vgg.trainable = False
    return vgg

def run_style_transfer(content_image, style_image):
    pass

