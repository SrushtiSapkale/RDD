# data.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(batch_size=32, img_size=(299, 299)):
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        's3://mydatasetbucket23/DATASET/TRAIN',  # Replace with actual path
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    # Validation data generator
    validation_generator = val_datagen.flow_from_directory(
        's3://mydatasetbucket23/DATASET/VALIDATE',  # Replace with actual path
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator

# Define create_generators function for convenience
def create_generators(batch_size=32, img_size=(299, 299)):
    return get_data_generators(batch_size, img_size)

