#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


def next_batch(x_train, y_train, batch_size, index_in_epoch, num_examples):
    """Return the next `batch_size` examples from this data set."""
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_examples:
      # Shuffle the data
      perm = numpy.arange(num_examples)
      numpy.random.shuffle(perm)
      x_train = x_train[perm]
      y_train = y_train[perm]
      # Start next epoch
      start = 0
      index_in_epoch = batch_size
      assert batch_size <= num_examples
    end = index_in_epoch
    return x_train[start:end], y_train[start:end], index_in_epoch




print "Reading data"
data = genfromtxt('bmi_train.csv',delimiter=',', filling_values=0)  # Training data
test_data = genfromtxt('bmi_test.csv',delimiter=',', filling_values=0)  # Test data
print "Processing data"
x_train=np.array([ i[:-1] for i in data])
y_train=np.array([i[-1] for i in data])

x_test=np.array([ i[:-1] for i in test_data])
y_test = np.array([i[-1] for i in test_data])


#  A number of features, 4 in this example
#  B = 3 species of Iris (setosa, virginica and versicolor)
A=data.shape[1]-1 # Number of features, Note first is y
B = 1
#tf_in = tf.placeholder("float", [None, A]) # Features
#tf_weight = tf.Variable(tf.random_normal([A,B], stddev=0.35), name="weights")
#tf_bias = tf.Variable(tf.zeros([B]))
#tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)

# Parameters
learning_rate = 0.001

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = A
n_classes = B
batch_size = 100
training_epochs = 15
display_step = 1
num_examples = len(x_train)
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")

weights = {
    'h1': tf.Variable(tf.random_normal([A, n_hidden_1], stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2,B], stddev=0.01), name="weights")
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

pred = multilayer_perceptron(x, weights, biases)
# Training via backpropagation
#tf_softmax_correct = tf.placeholder("float", [None,B])
#tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))

# Train using tf.train.GradientDescentOptimizer
#tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)

# Add accuracy checking nodes
#tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
#tf.accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))
cost = tf.reduce_sum(tf.pow(pred-y,2))/(2*num_examples)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Initialize and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

total_batches = num_examples/batch_size
batches_plt = np.arange(total_batches)
epochs_plt = np.arange(training_epochs)
cost_plt = np.zeros((total_batches,))
avg_cost_plt= np.zeros((training_epochs,))
print("...")
# Run the training
for epoch in range(training_epochs):
  idx = 0
  avg_cost = 0
  for i in range(total_batches):
    #sess.run(tf_train_step, feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})
    x_batch, y_batch, idx = next_batch(x_train, y_train, batch_size, idx,  num_examples)  
    _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
    avg_cost = c/total_batches
    cost_plt[i] = c + cost_plt[i]
  avg_cost_plt[epoch] = avg_cost
  if epoch % display_step == 0:
    print "Epoch:", '%d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
print "Optimization Finished"
# Print accuracy
    #result = sess.run(tf_accuracy, feed_dict={tf_in: x_test, tf_softmax_correct: y_test_onehot})
    #print "Run {},{}".format(i,result)

# Calculate accuracy
print("Testing... (Mean square loss Comparison)")
testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * x_test.shape[0]),feed_dict={x: x_test, y: y_test})  # same function as cost above
print("Testing cost=", testing_cost)



cost_plt /= num_examples
plt.subplot(221)
axes_1 = plt.gca()
axes_1.set_ylim([0,0.1])
plt.plot(batches_plt, cost_plt, 'ro')
plt.ylabel("Average Cost")
plt.xlabel("Batch")
plt.title("Batch Cost")

plt.subplot(222)
axes_2 = plt.gca()
axes_2.set_ylim([0,0.1])
plt.plot(epochs_plt, avg_cost_plt, 'bo')
plt.ylabel("Average Cost")
plt.xlabel("Epochs")
plt.title("Epoch Cost")

plt.subplot(223)
axes_3 = plt.gca()
axes_3.set_ylim([-10,70])
true = plt.scatter(x_test.sum(axis=1), y_test, marker = 'x', color='b')
pred = plt.scatter(x_test.sum(axis=1), sess.run(pred, feed_dict={x: x_test, y:y_test}), marker = 'x', color ='r')
plt.ylabel("BMI")
plt.xlabel("Input")
plt.title("BMI Prediction")
plt.legend((true, pred), ('Original Data', 'Predicted Data'),scatterpoints=1,loc='upper right')


plt.show()




