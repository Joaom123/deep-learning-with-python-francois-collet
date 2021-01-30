from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

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

# preparing the data - transform into a float32 array of shape (60000, 28 * 28) with values between 0 and 1
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# preparing the labels - categorically encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# training the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# get the accuracy using the test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
