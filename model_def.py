import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.accuracy import Accuracy
from tensorflow.keras.optimizers import Adam
import determined.keras as det
import data
from data import create_generators
import os


# You can replace this with a mock context for local testing
class MockContext:
    def get_hparam(self, param_name):
        # Provide default values for hyperparameters during local testing
        if param_name == "batch_size":
            return 32
        elif param_name == "learning_rate":
            return 1e-4
        elif param_name == "epochs":
            return 10
        return None


class RetinalDetachmentModel(det.TFKerasTrial):
    def __init__(self, context=None):
        # If running locally, mock the context with default values
        if context is None:
            context = MockContext()

        self.context = context
        self.img_size = (299, 299)
        self.batch_size = context.get_hparam("batch_size")
        self.learning_rate = context.get_hparam("learning_rate")
        self.epochs = context.get_hparam("epochs")

    def build_model(self):
        # Load base model
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        base_model.trainable = False  # Freeze layers

        # Add custom layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)

        # Outputs
        classification_output = Dense(1, activation='sigmoid', name='classification')(x)
        bbox_output = Dense(4, activation='linear', name='bounding_box')(x)




    def binary_crossentropy_loss(y_true, y_pred):
        """
        Calculates the binary cross-entropy loss for binary classification.

        Args:
            y_true: Tensor of true labels.
            y_pred: Tensor of predicted probabilities.

        Returns:
            Binary cross-entropy loss.
        """
    # Ensure the inputs are TensorFlow tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Calculate binary cross-entropy loss
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)


    # Compile model
    model = Model(inputs=base_model.input, outputs=[classification_output, bbox_output])
    model.compile(
        optimizer=Adam(learning_rate=self.learning_rate),
        loss={'classification': 'loss'},
        metrics={'classification': 'Accuracy'}
    )

    return model

def build_training_data_loader(self):
    train_generator, _ = data.get_data_generators(self.batch_size, self.img_size)
    return train_generator

def build_validation_data_loader(self):
    _, val_generator = data.get_data_generators(self.batch_size, self.img_size)
    return val_generator


if __name__ == "__main__":
    from data import create_generators

    # Initialize model
    model = RetinalDetachmentModel(None)  # Passing None will use the mock context

    # Load data
    train_generator, val_generator = model.build_training_data_loader(), model.build_validation_data_loader()

    # Build the model
    model_instance = model.build_model()

    # Train the model (you can adjust epochs, steps as needed)
    model_instance.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1,  # Just a quick test with 1 epoch
        steps_per_epoch=1,  # Limit to 1 batch for quick validation
        validation_steps=1,
        verbose=1
    )

    # Save the model after training    
    model_path = 'retinal_detachment_detection_model.h5'  # Save directly in the current directory
    model_instance.save(model_path)
    print(f"Model saved at {model_path}")

