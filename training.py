import numpy as np
import leaf_reader
import tensorflow as tf
import models




train_data,train_labels,test_data,test_labels,train_descriptors,test_descriptors=leaf_reader.readTrainingData(update_descriptors = False)


#print(train_descriptors[0])
#train_data[:,0:64] = np.copy(train_descriptors)
#test_data[:,0:64] = np.copy(test_descriptors)
#shape check
#print(train_data.shape)
#print(train_labels.shape)
#print(test_data.shape)
#print(test_labels.shape)

# Parameters

#decaying learning rate
starter_learning_rate = 0.01
global_step = tf.Variable(0, trainable=False)

#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.99, staircase=True)
learning_rate = starter_learning_rate

training_epochs = 100

#how many times we go trough whole dataset in one epoch
iterations = 30

batch_size = 32

# how often (in epochs) evaluation happens
display_step = 1


#nn = models.ThreeLayerNet()
#nn = models.TwoLayerNet(400)
#nn = models.ConvNet1D(200,150)
#nn = models.ConvNet1D(256,150)
#nn = models.ConvNet1D2(128,64,180,stride1=1,stride2=1)
nn = models.FourierNet(128,200)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(nn.cost,global_step)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum =0.8).minimize(nn.cost,global_step)


is_correct_prediction = tf.equal(tf.argmax(nn.predict, 1), tf.argmax(nn.y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, "float"))

# Initializing the variables
init = tf.initialize_all_variables()

def rnd_shuffle():
    rng_state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(train_labels)
    np.random.set_state(rng_state)
    np.random.shuffle(train_descriptors)
    return


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    batch_count = int(train_data.shape[1]/batch_size)  

    # Training cycle
    for epoch in range(training_epochs):

        avg_cost = 0.

        for i in range(iterations):        
            rnd_shuffle()
            # Loop over all batches
            for j in range(batch_count):
                batch_x = train_data[j*batch_size:(j+1)*batch_size]
                batch_y = train_labels[j*batch_size:(j+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)

                if (isinstance(nn,models.FourierNet)):
                    _, c = sess.run([optimizer, nn.cost], 
                    feed_dict={nn.x1: batch_x[:,0:64],nn.x2:batch_x[:,64:192],nn.y: batch_y})
                else:
                     _, c = sess.run([optimizer, nn.cost], 
                    feed_dict={nn.x: batch_x,nn.y: batch_y})
                # Compute average loss
                avg_cost += c / (batch_count*iterations)

        # Display logs per epoch step
        if ((epoch+1) % display_step == 0):
            if (isinstance(nn,models.FourierNet)):
                acc=accuracy.eval({nn.x1: test_data[:,0:64], nn.x2: test_data[:,64:192], nn.y: test_labels})
            else:
                acc=accuracy.eval({nn.x: test_data, nn.y: test_labels})
            print("Epoch:", '%04d' % (epoch), "training cost=","{:.9f}".format(avg_cost), \
            "accuracy=","{:.9f}".format(acc))

    print("Optimization Finished!")