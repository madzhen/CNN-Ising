import tensorflow as tf
import numpy as np


batch_size=5000

step=2000
out_step=10
Train_Set=np.load("Train_Set_final_100tho.npy")
Train_label=np.load("label_Set_final_100tho.npy")


x = tf.placeholder(tf.float32, [None, 256])
y_ = tf.placeholder(tf.float32, [None, 100])

W=np.zeros((320,100),dtype=np.float32)

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')


def next_batch(i):
	return (Train_Set[i*batch_size:(i+1)*batch_size],Train_label[i*batch_size:(i+1)*batch_size])




input =tf.reshape(x, [-1, 16, 16, 1])


filter = weight_variable([3,3,1,5])
W_out = weight_variable([320, 100])



conv_out=tf.nn.relu(conv2d(input,filter))

keep_prob = tf.placeholder("float")
conv_out_drop = tf.reshape(tf.nn.dropout(conv_out, keep_prob),[-1,320])


y_conv=tf.nn.softmax(tf.matmul(conv_out_drop, W_out) )

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



sess.run(tf.initialize_all_variables())


for j in range(out_step):
	print(j)
	for i in range(step):
	    batch = next_batch(i)
	    if i % 100 == 0:
	        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
	        print("step %d, training accuracy %g" % (i, train_accuracy))
	    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})

	training_data = np.hstack((Train_Set, Train_label))
	np.random.shuffle(training_data)
	Train_Set = training_data[:,0:256]
	Train_label = training_data[:, 256:356]    

W=W_out.eval()
np.save("W.npy",W)