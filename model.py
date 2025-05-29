from tensorflow.keras import layers, models, applications

def build_generator():
    inputs = layers.Input(shape=(24, 24, 3))
    x = layers.Conv2D(64, 9, padding='same', activation='relu')(inputs)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
    model = models.Model(inputs, x)
    return model

def build_discriminator():
    inputs = layers.Input(shape=(96, 96, 3))
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, x)
    return model

def build_vgg(input_shape):
    vgg = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg.trainable = False
    model = models.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    return model