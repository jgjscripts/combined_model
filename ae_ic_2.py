import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Define the autoencoder model
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

# Define the classification model
def build_classification_model(input_shape, num_classes):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    classification_model = Model(input_img, output)
    return classification_model

# Define input shape and number of classes
input_shape = (240, 320, 3)  # Adjusted according to your image dimensions
num_classes = 10  # Adjust according to your classification task

# Build autoencoder model
autoencoder = build_autoencoder(input_shape)

# Build classification model
classification_model = build_classification_model(input_shape, num_classes)

# Get the output of the autoencoder
autoencoder_output = autoencoder.output

# Pass the output of the autoencoder as input to the classification model
classification_output = classification_model(autoencoder_output)

# Create the combined model
combined_model = Model(inputs=autoencoder.input, outputs=classification_output)

# Compile the combined model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the combined model
combined_model.summary()
