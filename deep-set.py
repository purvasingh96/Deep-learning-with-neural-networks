import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("tmp/data", one_hot=True)

n_nodes_hidden_layer_1 = 500
n_nodes_hidden_layer_2 = 500
n_nodes_hidden_layer_3 = 500

n_classes = 10
# feed data in form of batches
batch_size = 100

# matrix = height*width
# size of each data in 2-D array would be 28*28
# we are flattening data from 2D to 1D array
#so 28*28=784
# x is value of data
#y is label of data
x= tf.placeholder('float', [None, 784])
y= tf.placeholder('float')

def neural_network_model(data):
    # (input_data  * weights) + biases
    # if all the input_data is 0, then due to biases, atleast some neurons would fire signal
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hidden_layer_1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hidden_layer_1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_1, n_nodes_hidden_layer_2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer_2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_2, n_nodes_hidden_layer_3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer_3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data  * weights) + biases
    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']) ,hidden_layer_1['biases'])
    # since the matrix multiplication of data and weights can result in non-linearities
    # we add complexities to output values.
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']) , hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']) , hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.add(tf.matmul(layer_3, output_layer['weights']) , output_layer['biases'])
    output = tf.nn.relu(output)

    return output

def train_neural_network(input_data):
    prediction = neural_network_model(input_data)
    # tf.reduce_mean is same as numpy.mean(data, 1)
    # calculates difference between prediction that we got and know label (y) that we have
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
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