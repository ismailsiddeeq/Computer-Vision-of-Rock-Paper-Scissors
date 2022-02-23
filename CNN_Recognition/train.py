import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from kerastuner.tuners import RandomSearch
import imageio

# find datasets
tfds.list_builders()

# get info from data
builder = tfds.builder('rock_paper_scissors')
info = builder.info

# prepare rock, paper, scissor data
ds_train = tfds.load(name="rock_paper_scissors", split="train")
ds_test = tfds.load(name="rock_paper_scissors", split="test")

# data prep
train_images = np.array([example['image'].numpy()[:, :, 0] for example in ds_train])
train_labels = np.array([example['label'].numpy() for example in ds_train])
test_images = np.array([example['image'].numpy()[:, :, 0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])

train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255


# train network
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.AveragePooling2D(6, 3, input_shape=(300, 300, 1)))

    for i in range(hp.Int("Conv Layers", min_value=0, max_value=3)):
        model.add(keras.layers.Conv2D(hp.Choice(f"layer_{i}_filters",
                                                [16, 32, 64]), 3, activation='relu'))

    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Choice("Dense layer", [64, 128, 256, 512, 1024]),
                                 activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
)

tuner.search(train_images, train_labels,
             validation_data=(test_images, test_labels),
             epochs=10, batch_size=32)

best_model = tuner.get_best_models()[0]
best_model.evaluate(test_images, test_labels)
best_model.summary()
best_model.save('./model')


loaded_model = keras.models.load_model('./model')
loaded_model.evaluate(test_images, test_labels)


result = best_model.predict(np.array([train_images[0]]))
print(result)
print(train_images[0].shape)
predicted_value = np.argmax(result)
print(predicted_value)


im = imageio.imread('rock.jpg')
im_np = np.asarray(im)[:, :, 0]
im_np = im_np.reshape(-1, 300, 300, 1)

result = best_model.predict(np.array([im_np]))
print(result)
predicted_value = np.argmax(result)
print(predicted_value)



