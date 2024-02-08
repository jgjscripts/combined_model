import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

# Define the Autoencoder + CNN model
def build_autoencoder_cnn_model(input_shape, num_classes):
    # Define the Autoencoder part
    autoencoder_input = Input(shape=input_shape, name='autoencoder_input')
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(autoencoder_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    # Define the CNN part for decoded images
    cnn_input_decoded = Input(shape=input_shape, name='cnn_input_decoded')
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input_decoded)
    y = MaxPooling2D((2, 2), padding='same')(y)
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = MaxPooling2D((2, 2), padding='same')(y)
    y = Flatten()(y)

    # Define the CNN part for original images
    cnn_input_original = Input(shape=input_shape, name='cnn_input_original')
    z = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input_original)
    z = MaxPooling2D((2, 2), padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = MaxPooling2D((2, 2), padding='same')(z)
    z = Flatten()(z)

    # Combine outputs from both CNN models and the decoded images
    combined_output = concatenate([y, z])
    
    # Dense layers for combined output
    combined_output = Dense(128, activation='relu')(combined_output)
    output = Dense(num_classes, activation='sigmoid')(combined_output)
    
    # Define the combined model
    model = Model(inputs=[autoencoder_input, cnn_input_decoded, cnn_input_original], outputs=output)
    
    return model

# Define input shape and number of classes
input_shape = (240, 320, 3)  # Adjusted according to your image dimensions
num_classes = 1  # Binary classification

# Build the model
model = build_autoencoder_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
