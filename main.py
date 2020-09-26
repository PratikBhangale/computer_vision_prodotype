import random

import tensorflow as tf
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop
import zipfile
import os

# path = 'D:\\Coding Projects\\tensorflow_practice\\archive.zip'
# zipref1 = zipfile.ZipFile(path)
# try:
#     os.mkdir('D:\\Coding Projects\\Datasets')
# except OSError as e:
#     print(e)
# print('Starting Extraction')
# zipref1.extractall('D:\\Coding Projects\\Datasets')
# print('Extraction finished...')
# zipref1.close()

# try:
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs')
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs\\training')
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs\\testing')
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs\\training\\cats')
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs\\training\\dogs')
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs\\testing\\cats')
#     os.mkdir('D:\\Coding Projects\\Datasets\\cats_v_dogs\\testing\\dogs')
# except OSError as e:
#     print(e)
#
#
# def save_data(SOURCE, TRAINING, TESTING, SPLIT_NUM):
#     all_files = []
#
#     for file_name in os.listdir(SOURCE):
#         file_path = os.path.join(SOURCE, file_name)
#
#         if os.path.getsize(file_path):
#             all_files.append(file_name)
#         else:
#             print('{} is zero length, so ignoring'.format(file_name))
#
#     n_files = len(all_files)
#     split_point = int(n_files * SPLIT_NUM)
#
#     shuffled = random.sample(all_files, n_files)
#
#     train_set = shuffled[:split_point]
#     test_set = shuffled[split_point:]
#
#     for file_name in train_set:
#         copyfile(os.path.join(SOURCE, file_name), os.path.join(TRAINING, file_name))
#
#     for file_name in test_set:
#         copyfile(os.path.join(SOURCE, file_name), os.path.join(TESTING, file_name))
#
#
training_cats_dir = 'D:\\Coding Projects\\Datasets\\cats_v_dogs\\training\\cats'
training_dogs_dir = 'D:\\Coding Projects\\Datasets\\cats_v_dogs\\training\\dogs'
testing_cats_dir = 'D:\\Coding Projects\\Datasets\\cats_v_dogs\\testing\\cats'
testing_dogs_dir = 'D:\\Coding Projects\\Datasets\\cats_v_dogs\\testing\\dogs'
cat_source_dir = 'D:\\Coding Projects\\Datasets\\PetImages\\Cat'
dog_source_dir = 'D:\\Coding Projects\\Datasets\\PetImages\\Dog'
split_ratio = 0.9


#
# print('Copying data...')
# save_data(cat_source_dir, training_cats_dir, testing_cats_dir, split_ratio)
# save_data(dog_source_dir, training_dogs_dir, testing_dogs_dir, split_ratio)
# print('Copying Finished.')
#
# print(len(os.listdir(training_cats_dir)))
# print(len(os.listdir(training_dogs_dir)))
# print(len(os.listdir(testing_cats_dir)))
# print(len(os.listdir(testing_dogs_dir)))

# working on tensorflow

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > 0.99:
            print('Reached 99 percent accuracy, stopping training.')
            self.model.stop_training = True


callbacks = myCallback()

training_dir = 'D:\\Coding Projects\\Datasets\\cats_v_dogs\\training'
train_datagen = ImageDataGenerator(rescale=1 / 255)
training_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary'
)

testing_dir = 'D:\\Coding Projects\\Datasets\\cats_v_dogs\\testing'
test_datagen = ImageDataGenerator(rescale=1 / 255)
testing_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # softmax
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])

model.fit(
    training_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=testing_generator,
    validation_steps=25
    # ,callbacks=[callbacks]
)
