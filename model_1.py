import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
import numpy as np

def gan_model(input_shape):
    # Define the generator (deblur GAN)
    input_gan = Input(shape=input_shape, name='input_gan')
    generator = Conv2D(32, (3, 3), activation='relu', padding='same')(input_gan)
    # Add more layers for your GAN
    output_gan = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(generator)
    
    # Define the GAN model
    gan_model = Model(inputs=input_gan, outputs=output_gan)
    
    return gan_model

def classification_model(input_shape, num_classes):
    # Define the image classification model
    input_classification = Input(shape=input_shape, name='input_classification')
    classification_model = Conv2D(32, (3, 3), activation='relu', padding='same')(input_classification)
    classification_model = MaxPooling2D((2, 2))(classification_model)
    # Add more layers for classification if needed
    classification_model = Flatten()(classification_model)
    classification_model = Dense(128, activation='relu')(classification_model)
    output_classification = Dense(num_classes, activation='softmax')(classification_model)
    
    # Define the classification model
    classification_model = Model(inputs=input_classification, outputs=output_classification)
    
    return classification_model

def combined_model(input_shape_gan, input_shape_classification, num_classes):
    # Create instances of GAN and classification models
    gan = gan_model(input_shape_gan)
    classification = classification_model(input_shape_classification, num_classes)
    
    # Concatenate outputs from both models
    gan_output = gan.output
    classification_output = classification.output
    combined_output = concatenate([gan_output, classification_output])
    
    # Define final classification layer for binary classification
    combined_output = Dense(2, activation='softmax', name='output_combined')(combined_output)
    
    # Define the combined model
    combined_model = Model(inputs=[gan.input, classification.input], outputs=combined_output)
    
    return combined_model

# Define input shapes and number of classes
input_shape = (240, 320, 3)  # Adjusted according to your image dimensions
num_classes = 2  # Binary classification

# Create an instance of the combined model
model = combined_model(input_shape, input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Generate dummy data for training and testing
x_train_gan = np.random.rand(100, 240, 320, 3)
x_train_classification = np.random.rand(100, 240, 320, 3)
y_train = np.random.randint(0, 2, size=(100,))

x_test_gan = np.random.rand(20, 240, 320, 3)
x_test_classification = np.random.rand(20, 240, 320, 3)
y_test = np.random.randint(0, 2, size=(20,))

# Train the model
model.fit([x_train_gan, x_train_classification], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate([x_test_gan, x_test_classification], y_test)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = model.predict([x_test_gan, x_test_classification])
print("Predictions:", predictions)
