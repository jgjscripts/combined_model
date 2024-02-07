import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

def build_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape, name='input_image')
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded, name='autoencoder')
    return autoencoder

def build_classification_model(input_shape, num_classes):
    input_img = Input(shape=input_shape, name='input_image')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_classification = Dense(num_classes, activation='softmax', name='classification_output')(x)

    classification_model = Model(input_img, output_classification, name='classification_model')
    return classification_model

# Define input shape and number of classes
input_shape = (240, 320, 3)  # Adjusted according to your image dimensions
num_classes = 2  # Binary classification

# Build autoencoder model
autoencoder = build_autoencoder(input_shape)

# Build image classification model
classification_model = build_classification_model(input_shape, num_classes)

# Combine autoencoder output with classification model input
combined_input = autoencoder.input
classification_input = autoencoder.output
classification_output = classification_model(classification_input)

# Print shapes before concatenation
print("Shape of classification layer output:", classification_model.layers[-2].output.shape)
print("Shape of classification output:", classification_output.shape)

# Concatenate dense layers' output
classification_layer = classification_model.layers[-2].output
combined_layer = concatenate([classification_layer, classification_output])

# Output layer
combined_output = Dense(num_classes, activation='softmax', name='combined_output')(combined_layer)

# Define the combined model
combined_model = Model(inputs=combined_input, outputs=combined_output)

# Compile the model
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
combined_model.summary()
