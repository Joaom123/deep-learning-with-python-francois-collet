from keras import models, layers
from keras.datasets import mnist

# the images and labels have one-to-one correspondence
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# see the data we dealing with
# print(train_images.shape)
# print(train_labels)  # 0 to 9
# print(train_labels.shape)

# the network architecture
network = models.Sequential()
network.add(
    layers.Dense(
        512,
        activation='relu',
        input_shape=(28 * 28,)
    )
)
network.add(
    layers.Dense(
        10,
        activation='softmax'
    )
)

# the compilation step
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)