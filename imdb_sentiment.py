import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import codecs
import matplotlib.pyplot as plt
import re
from random import randint

#GloVe Path
glove_path = "glove.6B.50d.txt"

words_list, word_vector = [], []
with codecs.open(glove_path, 'r', encoding='utf-8') as glove_file:
	for line in glove_file.readlines():
		row = line.strip().split()
		words_list.append(row[0])
		x = [float(val) for val in row[1:]]
		word_vector.append(x)

	print "Loaded GloVe!!!"

word_vectors = np.array(word_vector, dtype='float32')
print "GloVe loading Done (Numpy Array)!!!"
print len(words_list)
print len(word_vector)
print word_vectors.shape

positive_files = ["training_data/positiveReviews/" + f for f in listdir("training_data/positiveReviews/") if isfile(join("training_data/positiveReviews/", f))]
negative_files = ["training_data/negativeReviews/" + f for f in listdir("training_data/negativeReviews/") if isfile(join("training_data/negativeReviews/", f))]

max_seq_length = 250

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_sentences(text):
	text = text.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", text.lower())

ids = np.zeros((25000, max_seq_length), dtype='int32')
file_counter = 0
for pf in positive_files:
	with codecs.open(pf, 'r', encoding='utf-8') as f:
		index_counter = 0
		line = f.readline()
		cleaned_line = clean_sentences(line)
		text = cleaned_line.split()
		
		for word in text:
			try:
				ids[file_counter][index_counter] = words_list.index(word)
			except ValueError:
				ids[file_counter][index_counter] = 399999 #Vector for unknown words

			index_counter += 1
			if index_counter >= max_seq_length:
				break
		
		file_counter += 1

print "Positive files done!!!"

for nf in negative_files:
	with codecs.open(nf, 'r', encoding='utf-8') as f:
		line = f.readline()
		index_counter = 0
		line = f.readline()
		cleaned_line = clean_sentences(line)
		text = cleaned_line.split()
		
		for word in text:
			try:
				ids[file_counter][index_counter] = words_list.index(word)
			except ValueError:
				ids[file_counter][index_counter] = 399999 #Vector for unknown words

			index_counter += 1
			if index_counter >= max_seq_length:
				break
		
		file_counter += 1

print "Negative files done!!!"

np.save('ids_matrix', ids)
print "ids matrix created!!!"

#ids = np.load('ids_matrix.npy')

def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

batch_size = 24
lstm_units = 64
num_classes = 2
num_dimensions = 50
iterations = 50000

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])

data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors,input_data)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.5)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
	#Next Batch of reviews
	next_batch, next_batch_labels = get_train_batch();
	sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})
	print "Iteration : ", i
	print("Accuracy for this batch:", (sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)

	#Save the network every 10,000 training iterations
	if (i % 10000 == 0 and i != 0):
		save_path = saver.save(sess, "trained_models/lstm_model.ckpt", global_step=i)
		print("saved to %s" % save_path)

#Testing the model on test data
saver.restore(sess, tf.train.latest_checkpoint("trained_models/"))

iterations = 10
for i in range(iterations):
    next_batch, next_batch_labels = get_test_batch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)

