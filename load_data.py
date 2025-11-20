import tensorflow as tf

dataset_path = 'soil-data'  # path to your dataset folder

# Load training dataset (80%) and validation dataset (20%)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),  # Resize images to 224x224
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

print("Number of batches in training dataset:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Number of batches in validation dataset:", tf.data.experimental.cardinality(val_dataset).numpy())
