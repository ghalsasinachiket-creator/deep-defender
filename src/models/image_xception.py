import tensorflow as tf

def build_xception_classifier(input_shape=(299, 299, 3), trainable=False):
    """
    Builds Xception-based binary classifier.
    Returns a tf.keras.Model.
    """
    base = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    base.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="xception_deepfake")
    return model

def load_xception_weights(model, weights_path):
    """
    Loads trained weights into the model.
    """
    model.load_weights(weights_path)
    return model