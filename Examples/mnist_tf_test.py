import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from icecream import ic

# data is stored in 1 file, for large datasets w multiple files on disk it's good practice to shuffle them when traning
# as_supervised=True=>will return tubple object (img, label) instead of dict {'image':img, 'label':label}
(mnist_train, mnist_test), mnist_info = tfds.load('mnist',
                                                  split=['train', 'test'],
                                                  shuffle_files=True,
                                                  as_supervised=True,
                                                  with_info=True)

#--------------------------------------------------------
#                 Training Pipeline
#--------------------------------------------------------
def normalise_img(image, label):
    """Normalise images `uint8`->`float32`."""
    return tf.cast(image, tf.float32) / 255.0, label # tf.cast casts a tensor to a new datatype

#TFDS privides images of type tf.uint8, while the model will need tf.float32, so we cast & normalise them
mnist_train = mnist_train.map(normalise_img, num_parallel_calls=tf.data.AUTOTUNE)
mnist_train = tf.reshape(mnist_train, [28*28])
# This will improve performance, As you fit the dataset in memory, cache it before shuffling.
mnist_train = mnist_train.cache()
# We want true randomness, we set the shuffle buffer to the full dataset size
# If the dataset can't fit in memory use buffer_size=1000 (if the system allows it)
mnist_train = mnist_train.shuffle(mnist_info.splits['train'].num_examples) 
# Each Epoch won't be the entire dataset, instead we split it into unique batches of 128 images
mnist_train = mnist_train.batch(128)
# It's good practice to end a pipeline by prefetching for performance
mnist_train = mnist_train.prefetch(tf.data.AUTOTUNE)

#--------------------------------------------------------
#                 Testing Pipeline
#--------------------------------------------------------
mnist_test = mnist_test.map(normalise_img, num_parallel_calls=tf.data.AUTOTUNE)
mnist_test = tf.reshape(mnist_test, [28*28])
mnist_test = mnist_test.batch(128)
mnist_test = mnist_test.cache()
mnist_test = mnist_test.prefetch(tf.data.AUTOTUNE)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),# Create a flatten layer
    keras.layers.Dense(128, activation='relu'),# with dense layer with 128 dimensions at the output
    keras.layers.Dense(10, activation='softmax') # Create the final dense layer that takes in 128 dimension input, and outputs label
])
model.compile(
    optimizer=keras.optimizers.Adam(0.001), # The learning rate is 0.001
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # This is the best for classification using 1-hot
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.fit(
    mnist_train,
    epochs=10,
    validation_data=mnist_test
)
