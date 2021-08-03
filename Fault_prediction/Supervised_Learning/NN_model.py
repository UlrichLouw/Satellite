import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print(tf.__version__)

X, Y = make_classification(n_samples = 50000, n_features = 10, n_informative = 8, n_redundant = 0, n_clusters_per_class = 2)
# we leave the n_classes parameter blank, by default it is 2

# One column corresponding to each class
Y = np.array([Y, -(Y-1)]).T

X, X_test, Y, Y_test = train_test_split(X,Y)

# Parameters

learning_rate = 0.001
training_epochs = 100
batch_size = 100

display_step = 1

# Neural Networl Parameters
n_hidden_1 = 10
n_hidden_2 = 10
# number of features
n_input = 10
# number of classes
n_classes = 2

x = tf.compat.v1.placeholder(n_input)
y = n_classes

# model
def multilayer_perceptron(x, weights, biases):
    # Hidden Layer with ReLU activation
    # ReLU - Acronym for Rectified Leaky Unit
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with ReLU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers' weight & bias values
weights = {
    'h1': tf.Variable(tf.compat.v1.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.compat.v1.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.compat.v1.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.compat.v1.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.compat.v1.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define losses and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # run over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run backpropagation
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch%display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    
    print("Optimization finished.")
        
        
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result 
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})