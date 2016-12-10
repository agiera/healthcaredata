#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Convert to one hot
def convertOneHot(data):
    y=np.array([int(i[-1]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max()+1)
        y_onehot[i][j]=1
    return (y,y_onehot)

def convertScalar(data):
  y = np.zeros((len(data), )) 
  for i in range(len(data)):
    prob = -1000000.0
    correct = 0
    for j in range(len(data[i])):
      if data[i][j]  > prob:
        prob = data[i][j]
        correct = j
    y[i] = correct
  return y 

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
data = genfromtxt('raceax_train.csv',delimiter=',', filling_values=0)  # Training data
test_data = genfromtxt('raceax_test.csv',delimiter=',', filling_values=0)  # Test data
print "Processing data"
x_train=np.array([ i[:-1] for i in data])
y_train,y_train_onehot = convertOneHot(data)

x_test=np.array([ i[:-1] for i in test_data])
y_test,y_test_onehot = convertOneHot(test_data)


#  A number of features, 4 in this example
#  B = 3 species of Iris (setosa, virginica and versicolor)
A=data.shape[1]-1 # Number of features, Note first is y
B=len(y_train_onehot[0])
#tf_in = tf.placeholder("float", [None, A]) # Features
#tf_weight = tf.Variable(tf.random_normal([A,B], stddev=0.35), name="weights")
#tf_bias = tf.Variable(tf.zeros([B]))
#tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)

# Parameters
learning_rate = 0.001

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 512
n_hidden_4 = 512
n_input = A
n_classes = B
batch_size = 100
training_epochs = 15
display_step = 1
num_examples = len(x_train)
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([A, n_hidden_1], stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.01)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_4,B], stddev=0.01), name="weights")
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
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
    x_batch, y_batch, idx = next_batch(x_train, y_train_onehot, batch_size, idx,  num_examples)  
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

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "Accuracy:", sess.run(accuracy, feed_dict=({x: x_test, y: y_test_onehot}))

cost_plt /= num_examples
plt.subplot(221)
axes_1 = plt.gca()
axes_1.set_ylim([0,0.001])
plt.plot(batches_plt, cost_plt, 'ro')
plt.ylabel("Average Cost")
plt.xlabel("Batch")
plt.title("Batch Cost")

plt.subplot(222)
axes_2 = plt.gca()
axes_2.set_ylim([0,0.001])
plt.plot(epochs_plt, avg_cost_plt, 'bo')
plt.ylabel("Average Cost")
plt.xlabel("Epoch")
plt.title("Epochs Cost")

plt.subplot(223)
true = plt.scatter(x_test.sum(axis=1), y_test, marker = 'o', color='b')
pred = plt.scatter(x_test.sum(axis=1), convertScalar(sess.run(pred, feed_dict={x: x_test, y:y_test_onehot})), marker = 'x', color ='r')
plt.ylabel("Race")
plt.xlabel("Input")
plt.title("Race Prediction")
plt.legend((true, pred), ('Original Data', 'Predicted Data'),scatterpoints=1,loc='upper right')
plt.show()


