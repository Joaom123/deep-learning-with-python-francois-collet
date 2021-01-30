from keras import models
from keras import layers

# layer compatibility -> every layer will only accept input tensors of
# a certain shape and will return output tensors of a certain shape

# this layer will only accept as input 2D tensors. It will return a tensor where the first
# dimension has been transformed to be 32
layer = layers.Dense(32, input_shape=(784,))

model = models.Sequential()
model.add(layers.Dense(
    32, input_shape=(784,)
))
model.add(layers.Dense(32))  # didn't receive an input shape - it automatically inferred its input
