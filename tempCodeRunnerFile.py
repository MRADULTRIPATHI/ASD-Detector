import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# import os

# # Set dataset path
# train_dir = 'dataset/train'
# val_dir = 'dataset/val'
# img_size = (128, 128)  # Resize frames
# batch_size = 32

# # Load and preprocess data
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     train_dir,
#     image_size=img_size,
#     batch_size=batch_size,
#     shuffle=True
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     val_dir,
#     image_size=img_size,
#     batch_size=batch_size,
#     shuffle=True
# )

# # Get class names
# class_names = train_ds.class_names
# print("Classes:", class_names)

# # Improve performance
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# # CNN model
# model = models.Sequential([
#     layers.Rescaling(1./255, input_shape=(128, 128, 3)),
#     layers.Conv2D(32, (3,3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(len(class_names), activation='softmax')
# ])

# # Compile model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train model
# history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# # Save model
# model.save('autism_behavior_cnn_model.h5')