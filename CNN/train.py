import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import datetime

# find datasets
tfds.list_builders()

# get info from data and prepare rock, paper, scissor data
ds = 'rock_paper_scissors'
(ds_train_raw, ds_test_raw), ds_info = tfds.load(name=ds,
                                                 data_dir='tmp',
                                                 with_info=True,
                                                 as_supervised=True,
                                                 split=[tfds.Split.TRAIN, tfds.Split.TEST], )

num_train = ds_info.splits['train'].num_examples
num_test = ds_info.splits['test'].num_examples
num_class = ds_info.features['label'].num_classes

img_size = ds_info.features['image'].shape[0]
img_shape = ds_info.features['image'].shape

img_size = img_size // 2
img_shape = (img_size, img_size, img_shape[2])


# data prep
def format_img(img, label):
    img = tf.cast(img, tf.float32)
    img = img / 255.
    img = tf.image.resize(img, [img_size, img_size])
    return img, label


# rotate the image by 0, 90, 180, 270 degree
def aug_rotation(img: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


# flip the image
def aug_flip(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


# change the image's color, brightness, contrast...
def aug_color(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_hue(img, max_delta=0.08)
    img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, lower=0.8, upper=1)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
    return img


# invert image's color
def aug_inversion(img: tf.Tensor) -> tf.Tensor:
    random = tf.random.uniform(shape=[], minval=0, maxval=1)
    if random > 0.5:
        img = tf.math.multiply(img, -1)
        img = tf.math.add(img, 1)
    return img


# zoom and crop the image
def aug_zoom(img: tf.Tensor, min_zoom=0.8, max_zoom=1.0) -> tf.Tensor:
    img_width, img_height, img_colors = img.shape
    crop_size = (img_width, img_height)
    scales = list(np.arange(min_zoom, max_zoom, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def img_crop(img):
        crops = tf.image.crop_and_resize([img],
                                         boxes=boxes,
                                         box_indices=np.zeros(len(scales)),
                                         crop_size=crop_size)
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    return tf.cond(choice < 0.5, lambda: img, lambda: img_crop(img))


# apply functions to augment the data to avoid overfitting
def aug_data(img, label):
    img = aug_rotation(img)
    img = aug_flip(img)
    img = aug_color(img)
    img = aug_inversion(img)
    img = aug_zoom(img)
    return img, label


# prepare dataset images
ds_train = ds_train_raw.map(format_img)
ds_test = ds_test_raw.map(format_img)

ds_train_aug = ds_train.map(aug_data)
batch_size = 32

# shuffle and batch the data
ds_train_aug_shuf = ds_train_aug.shuffle(buffer_size=num_train)
ds_train_aug_shuf = ds_train_aug.batch(batch_size=batch_size)
ds_train_aug_shuf = ds_train_aug_shuf.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds_test_shuf = ds_test.batch(batch_size)

# create cnn model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Convolution2D(input_shape=img_shape, filters=64, kernel_size=3,
                                        activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=3, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=3, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=num_class, activation=tf.keras.activations.softmax))

# display model's content
model.summary()

# compile and train
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x=ds_train_aug_shuf.repeat(),
          validation_data=ds_test_shuf.repeat(),
          epochs=15,
          steps_per_epoch=num_train // batch_size,
          validation_steps=num_test // batch_size,
          callbacks=[tf.keras.callbacks.TensorBoard(
              log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), histogram_freq=1)],
          verbose=1)

# print out the loss and accuracy of the model using the training data and testing data
train_loss, train_accuracy = model.evaluate(x=ds_train.batch(batch_size).take(num_train))
test_loss, test_accuracy = model.evaluate(x=ds_test.batch(batch_size).take(num_test))

# save the model
model.save('RPS_cnn.h5', save_format='h5')
