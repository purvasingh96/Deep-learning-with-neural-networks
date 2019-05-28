import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell


mnist = input_data.read_data_sets("tmp/data", one_hot=True)

n_classes = 10
no_of_epochs = 10
# feed data in form of batches
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

# matrix = height*width
# size of each data in 2-D array would be 28*28
# we are flattening data from 2D to 1D array
#so 28*28=784
# x is value of data
#y is label of data
x= tf.placeholder('float', [None, n_chunks, chunk_size])
y= tf.placeholder('float')

def recurrent_neural_network_model(data):
    # (input_data  * weights) + biases
    # if all the input_data is 0, then due to biases, atleast some neurons would fire signal
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    data = tf.transpose(data, [1, 0, 2])
    # -1 can be referred to as data's size
    data = tf.reshape(x, [-1, chunk_size])
    data = tf.split(0, n_chunks, data)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, state = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)


    output = tf.add(tf.matmul(outputs[-1], layer['weights']) , layer['biases'])
    output = tf.nn.relu(output)

    return output

def train_neural_network(input_data):
    prediction = recurrent_neural_network_model(input_data)
    # tf.reduce_mean is same as numpy.mean(data, 1)
    # calculates difference between prediction that we got and know label (y) that we have
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimizer will minimize the cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)



    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(no_of_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # if not using tensorflow
                # create own method to train batch_data
                epoc_x, epoch_y = mnist.train.next_batch(batch_size)
                epoc_x = epoc_x.reshape((batch_size, n_chunks, chunk_size))
                _, costValue = sess.run([optimizer, cost], feed_dict={x: epoc_x, y: epoch_y})
                epoch_loss+=costValue
                print('cost value :: ',costValue)
                print('Epoch ',epoch, 'completed out of ',no_of_epochs, ' loss is ',epoch_loss)

        correct = tf.equal(tf.arg_max(prediction,1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # evaluates how many predictions we made, that actually matched the labels
        print('Accuracy :: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))



train_neural_network(x)
