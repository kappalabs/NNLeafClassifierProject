import numpy as np
import leaf_reader
import tensorflow as tf
import models




train_data,train_labels,test_data,test_labels,train_descriptors,test_descriptors=leaf_reader.readTrainingData(update_descriptors = False)

#shape check
#print(train_data.shape)
#print(train_labels.shape)
#print(test_data.shape)
#print(test_labels.shape)

# Parameters

#decaying learning rate
starter_learning_rate = 0.01
global_step = tf.Variable(0, trainable=False)

#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.99, staircase=True)
learning_rate = starter_learning_rate

training_epochs = 100

#how many times we go trough whole dataset in one epoch
iterations = 20

batch_size = 16

# how often (in epochs) evaluation happens
display_step = 1


#nn = models.ThreeLayerNet()
#nn = models.TwoLayerNet(200)

nn = models.ConvNet1D(80,120)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(nn.cost,global_step)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum =0.9).minimize(nn.cost,global_step)


is_correct_prediction = tf.equal(tf.argmax(nn.predict, 1), tf.argmax(nn.y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, "float"))

# Initializing the variables
init = tf.initialize_all_variables()

def rnd_shuffle(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    batch_count = int(train_data.shape[1]/batch_size)

    rnd_shuffle(train_data, train_labels)

    # Training cycle
    for epoch in range(training_epochs):

        avg_cost = 0.

        for i in range(iterations):        
            # Loop over all batches
            for j in range(batch_count):
                batch_x = train_data[j*batch_size:(j+1)*batch_size] 
                batch_y = train_labels[j*batch_size:(j+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, nn.cost], feed_dict={nn.x: batch_x,nn.y: batch_y})
                # Compute average loss
                avg_cost += c / (batch_count*iterations)

        # Display logs per epoch step
        if ((epoch+1) % display_step == 0):
            acc=accuracy.eval({nn.x: test_data, nn.y: test_labels})
            print("Epoch:", '%04d' % (epoch), "training cost=","{:.9f}".format(avg_cost), \
            "accuracy=","{:.9f}".format(acc))

    print("Optimization Finished!")