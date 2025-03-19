import tensorflow as tf
# import tensorflow_hub as hub
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import os

print("Loading tabular data...")
tabular_df = pd.read_csv("/mnt/d/Project/Hairmony/sx_taxonomy.csv")
print("Tabular data loaded.")

BATCH_SIZE = 8  # Reduced batch size to prevent memory issues
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_DIR = "/mnt/d/Project/Hairmony/images"

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256)) / 255.0  # Reduced image size to 256x256
    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image

def create_dataset(tabular_df, batch_size=BATCH_SIZE):
    print("Creating dataset...")
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in tabular_df["image_name"]]
    categorical_columns = tabular_df.iloc[0].drop("image_name").index.tolist()
    labels = pd.get_dummies(tabular_df[categorical_columns])
    labels = labels.values.astype(np.float32)
    
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(lambda x: load_and_preprocess_image(x), num_parallel_calls=AUTOTUNE)
    image_dataset = image_dataset.cache()  # Cache images for faster loading
    image_dataset = image_dataset.map(lambda x: augment_image(x), num_parallel_calls=AUTOTUNE)
    
    tabular_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip(((image_dataset, tabular_dataset), tabular_dataset))  # (inputs, labels)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)  # Prefetch for efficiency
    
    print("Dataset created.")
    return dataset

print("Splitting dataset...")
data_size = len(tabular_df)
train_size = int(0.8 * data_size)

dataset = create_dataset(tabular_df)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
print("Dataset split complete.")

print("Building model...")
image_input = tf.keras.Input(shape=(256, 256, 3), name="image_input")  # Updated input size to match image preprocessing
image_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=image_input)  # Switched to MobileNetV2
image_features = layers.GlobalAveragePooling2D()(image_model.output)

num_tabular_features = dataset.element_spec[0][1].shape[-1]  # Get the correct number of features
tabular_input = tf.keras.Input(shape=(num_tabular_features,), name="tabular_input")
tabular_features = layers.Dense(128, activation='relu')(tabular_input)
tabular_features = layers.Dropout(0.3)(tabular_features)

merged = layers.Concatenate()([image_features, tabular_features])
merged = layers.Dense(256, activation='relu')(merged)
merged = layers.Dropout(0.3)(merged)
output = layers.Dense(num_tabular_features, activation='sigmoid', name="output_layer")(merged)

model = tf.keras.Model(inputs=[image_input, tabular_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("Model built.")
model.summary()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("hair_model.h5", save_best_only=True)

print("Starting training...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[checkpoint_cb])
print("Training complete.")
