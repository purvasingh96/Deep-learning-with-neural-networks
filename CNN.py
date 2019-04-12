import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("tmp/data", one_hot=True)


n_classes = 10
# feed data in form of batches
batch_size = 128

# matrix = height*width
# size of each data in 2-D array would be 28*28
# we are flattening data from 2D to 1D array
#so 28*28=784
# x is value of data
#y is label of data
# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])

# strides - movement of window
# ksize - size of window
# move 1 pixel at a time
def conv2d(data, weights):
    return tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME')

# move 2 pixels at a time
def maxpool2d(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolution_neural_network_model(data):
    # (input_data  * weights) + biases
    # if all the input_data is 0, then due to biases, atleast some neurons would fire signal
    # this is a 5*5 convolution, take 1 input and outputs 32 features
    # fully connected layer will compress the images
    weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
                      'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                      'W_fully_connected':tf.Variable(tf.random_normal([7*7*64, 1024])),
                      'output':tf.Variable(tf.random_normal([1024, n_classes]))
                      }

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                      'b_conv2': tf.Variable(tf.random_normal([64])),
                      'b_fully_connected':tf.Variable(tf.random_normal([1024])),
                      'output':tf.Variable(tf.random_normal([n_classes]))
                      }

    # reshaping data of size 784 to flat 28*28*1
    data = tf.reshape(data, shape=[-1, 28, 28, 1])

    ## activation function
    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fully_connected'])+biases['b_fully_connected'])

    output = tf.matmul(fc, weights['output'])+biases['output']

    return output

def train_neural_network(input_data):
    prediction = convolution_neural_network_model(input_data)
    # tf.reduce_mean is same as numpy.mean(data, 1)
    # calculates difference between prediction that we got and know label (y) that we have
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimizer will minimize the cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    no_of_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(no_of_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # if not using tensorflow
                # create own method to train batch_data
                epoc_x, epoch_y = mnist.train.next_batch(batch_size)
                _, costValue = sess.run([optimizer, cost], feed_dict={x: epoc_x, y: epoch_y})
                epoch_loss+=costValue
                print('cost value :: ',costValue)
                print('Epoch ',epoch, 'completed out of ',no_of_epochs, ' loss is ',epoch_loss)

        correct = tf.equal(tf.arg_max(prediction,1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # evaluates how many predictions we made, that actually matched the labels
        print('Accuracy :: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))



train_neural_network(x)