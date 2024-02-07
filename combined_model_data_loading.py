import tensorflow as tf

# Define the path to the directory containing your images
data_dir = '/path/to/your/directory'

# Define parameters for loading the dataset
batch_size = 32
img_height = 224
img_width = 224

# Use image_dataset_from_directory to load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',  # or 'categorical' if your labels are one-hot encoded
  color_mode='rgb',  # or 'grayscale' if your images are grayscale
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=True,
  seed=123,  # for reproducibility
  validation_split=0.2,  # adjust the split ratio if needed
  subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',  # or 'categorical' if your labels are one-hot encoded
  color_mode='rgb',  # or 'grayscale' if your images are grayscale
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=True,
  seed=123,  # for reproducibility
  validation_split=0.2,  # adjust the split ratio if needed
  subset='validation'
)

# Optionally, you can also configure prefetching for better performance
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

####################################################
def load_data(train_data_dir, val_data_dir, batch_size, img_height, img_width, seed=123):
    # Create ImageDataGenerator for data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load and preprocess the training dataset
    train_ds = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Assuming binary classification
        seed=seed
    )

    # Load and preprocess the validation dataset
    val_ds = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Assuming binary classification
        seed=seed
    )

    # Prefetch the datasets
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds

# Define paths to training and validation data directories
train_data_dir = '/path/to/training_data_directory'
val_data_dir = '/path/to/validation_data_directory'

# Load data
batch_size = 32
img_height = 240
img_width = 320
train_ds, val_ds = load_data(train_data_dir, val_data_dir, batch_size, img_height, img_width)

