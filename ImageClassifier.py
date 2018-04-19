import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import matplotlib.image as mpimg
from PIL import Image
from IPython.display import display
import numpy as np
import random
import time
import shutil
import sys, os
import hashlib

class ImageClassifier:
    good_path = None
    bad_path = None
    debug = False
    train_image_batch = None
    train_label_batch = None
    test_image_batch = None
    test_label_batch = None
    img_size = None
    sess = None
    
    # GENERAL VARS
    logs_path = 'CNN/log_stats_cnn'
    network_path = 'CNN/saved_cnn.ckpt'

    batch_size = None
    test_batch_percentage = 0.1
    learning_rate = 0.0001
    training_epochs = None
    training_epochs_log = 100

    # PARTICULAR VARS
    num_classes = 2
    max_pool_lost = 2
    
    def __init__(self, good_path, bad_path, img_size, \
                 batch_size=60, training_epochs=2000, test_batch_percentage=0.1, debug=False):
        self.good_path = good_path
        self.bad_path = bad_path
        self.debug = debug
        self.img_size = img_size
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.test_batch_percentage = test_batch_percentage
        
        # It can be inception without training
        if good_path == None or bad_path == None:
            return
        
        # create dataset arrays
        all_filepaths = [self.good_path+"/"+s for s in os.listdir(self.good_path)]
        all_labels = np.full((len(all_filepaths), 2), [1,0])
        all_filepaths = all_filepaths + [self.bad_path+"/"+s for s in os.listdir(self.bad_path)]
        all_labels = np.vstack((all_labels, np.full((len(all_filepaths)-len(all_labels), 2), [0,1])))
        
        # create a partition vector
        partitions = [0] * len(all_filepaths)
        partitions[:int(len(all_filepaths) * self.test_batch_percentage)] = [1] * int(len(all_filepaths) * self.test_batch_percentage)
        random.shuffle(partitions)
        
        # convert string into tensors
        all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
        all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)
        
        # partition our data into a test and train set according to our partition vector
        train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
        train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

        # create input queues
        train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=True)
        test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=True)

        # process path and string tensor into an image and a label
        file_content = tf.read_file(train_input_queue[0])
        train_image = tf.image.decode_jpeg(file_content, channels=3)
        train_label = train_input_queue[1]

        file_content = tf.read_file(test_input_queue[0])
        test_image = tf.image.decode_jpeg(file_content, channels=3)
        test_label = test_input_queue[1]

        # define tensor shape
        train_image.set_shape([img_size, img_size, 3])
        test_image.set_shape([img_size, img_size, 3])

        # collect batches of images before processing
        self.train_image_batch, self.train_label_batch = tf.train.batch([train_image, train_label], batch_size=self.batch_size)
        self.test_image_batch, self.test_label_batch = tf.train.batch([test_image, test_label], \
                                                          batch_size=int(len(all_filepaths) * self.test_batch_percentage))
        
    def train(self):
        with tf.device("/gpu:0"):
            # INPUTS AND DESIRED OUTPUT
            X = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3],name="input")
            Y = tf.placeholder(tf.float32, [None, self.num_classes], name="prediction")

            # WEIGHTS AND BIAS
            fc_final_img_size = int(self.img_size / self.max_pool_lost / self.max_pool_lost / self.max_pool_lost) # 3 Max Pool
            weights = {
                'wc1': self.init_weights([6, 6, 3, 32], "Weight_conv1"),   # 3x3x1 conv, 32 outputs
                'wc2': self.init_weights([4, 4, 32, 64], "Weight_conv2"),  # 3x3x32 conv, 64 outputs
                'wc3': self.init_weights([3, 3, 64, 128], "Weight_conv3"), # 3x3x32 conv, 128 outputs
                'wfc': self.init_weights([128 * fc_final_img_size * fc_final_img_size, 625], "Weight_FC"), # FC 128 * size * size inputs, 625 outputs
                'wsm': self.init_weights([625, self.num_classes], "Weight_Softmax")         # FC 625 inputs, num_classes outputs (labels)
            }

            biases = {
                'bc1': self.init_weights([32], "Bias_conv1"),
                'bc2': self.init_weights([64], "Bias_conv2"),
                'bc3': self.init_weights([128], "Bias_conv3"),
                'bfc': self.init_weights([625], "Bias_FC"),
                'bsm': self.init_weights([self.num_classes], "Bias_Softmax")
            }

            # MODEL
            p_keep_conv = tf.placeholder(tf.float32, name="p_keep_conv")
            p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")
            Y_ = self.model(X, weights, biases, p_keep_conv, p_keep_hidden)

            # LOSS FUNCTION AND OPTIMIZER
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Y)
            cost = tf.reduce_mean(cross_entropy)
            train_step  = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

            # DEFINE ACCURACY FOR SUMMARY
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # SUMMARIES
            tf.summary.scalar("cost", cost)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()
            
            # RUN
            config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
            with tf.Session(config = config) as sess:
                start_time = time.time()
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
                saver = tf.train.Saver()
                for epoch in range(self.training_epochs):                    
                    # Image and Label get Batches from pipeline
                    batch_images, batch_labels = sess.run([self.train_image_batch, self.train_label_batch])
                    valid_images, valid_labels = sess.run([self.test_image_batch, self.test_label_batch])
                    
                    _, summary = sess.run([train_step, summary_op], feed_dict={X: batch_images, Y: batch_labels,\
                                                      p_keep_conv: 0.8, p_keep_hidden: 0.7})
                    writer.add_summary(summary, epoch)

                    if epoch % self.training_epochs_log == 0:
                        print("Epoch: ", epoch)
                        saver.save(sess, self.logs_path + 'model.ckpt', global_step=epoch)
                        batch_accuracy = accuracy.eval(feed_dict={X: batch_images, Y: batch_labels, \
                                                                  p_keep_conv: 1.0, p_keep_hidden: 1.0})
                        validation_accuracy = accuracy.eval(feed_dict={X: valid_images, Y: valid_labels, \
                                                                  p_keep_conv: 1.0, p_keep_hidden: 1.0})
                        print("Accuracy for batch: %s / Accuracy for validation: %s" % (batch_accuracy, validation_accuracy))

                        # TIMING CONTROL
                        elapsed_time = time.time() - start_time
                        mins = int(elapsed_time / 60)
                        secs = elapsed_time - (mins * 60)
                        print("Accumulative time: %02d:%02d" % (mins, int(secs % 60)))
                        print("----------------------")

                validation_accuracy = accuracy.eval(feed_dict={X: valid_images, Y: valid_labels, p_keep_conv: 1.0, p_keep_hidden: 1.0})
                print("Final Accuracy:", validation_accuracy)

                # SAVE IT
                save_path = saver.save(sess, self.network_path)
                print("Model saved to %s" % save_path)
                print("Finished in %.2f seconds" % ((time.time() - start_time)))
                
                coord.request_stop()
                coord.join(threads)
                sess.close()
                
    def load(self, checkpoint_restore=0):
        if (checkpoint_restore != 0):
            self.network_path = logs_path + "model.ckpt-" + str(checkpoint_restore)
            print ("Checkpoint to load( %s ): " % (checkpoint_restore, network_path))

        self.sess = tf.InteractiveSession()  
        new_saver = tf.train.import_meta_graph(self.network_path + '.meta')
        new_saver.restore(self.sess, self.network_path)
        tf.get_default_graph().as_graph_def()

    def run(self, source_path, good_path, bad_path, batch_size=60, good_percent_treshold=50, delete_images=False):
        if not os.path.exists(source_path):
            raise ValueError('The source for already normalized images path not found.')
        if not os.path.exists(source_path):
            os.makedirs(source_path)
        if not os.path.exists(good_path):
            os.makedirs(good_path)
        if not os.path.exists(bad_path):
            os.makedirs(bad_path)
            
        x = self.sess.graph.get_tensor_by_name("input:0")
        y_conv = self.sess.graph.get_tensor_by_name("output:0")
            
        for image_path in os.listdir(source_path):
            filename = source_path +"/"+ image_path
            if os.path.isdir(filename):
                continue
                
            try:
                img = mpimg.imread(filename)
            except:
                print ("Unloadable image: " + filename)
                continue
                
            image_0 = np.resize(img,(1, self.img_size, self.img_size, 3))
            _, result = self.sess.run(["input:0", y_conv], feed_dict= {x:image_0, "p_keep_conv:0": 1.0, "p_keep_hidden:0": 1.0})
            is_good_image = (np.argmax(result[0]) == 0)
            result = self.sess.run(tf.nn.softmax(result))
            if self.debug:
                print ("Result is: ", "GOOD" if is_good_image else "BAD")
                print ("With a %.2f %%" % (result[0][np.argmax(result)] * 100))
                
            # GET THE CORRECT FILENAME
            percent = int(result[0][np.argmax(result)]*100)
            fname, fextension = os.path.splitext(image_path)
            filehash = self.file_hash(filename)
            final_filename = filehash +"_"+ str(percent) +"_GOOD"+ fextension
            if not is_good_image:
                final_filename = filehash +"_"+ str(percent) +"_BAD"+ fextension
            
            # MOVE TO CORRECT PATH
            final_path = bad_path +"/"+ final_filename
            if is_good_image and percent >= good_percent_treshold:
                final_path = good_path +"/"+ final_filename
                
            
            if delete_images:
                shutil.move(filename, final_path)
            else:
                shutil.copy(filename, final_path)
                
                
    def test(self):
        # RUN
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
        with tf.Session(config = config) as sess:
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            batch_images, batch_labels = sess.run([self.train_image_batch, self.train_label_batch])
            valid_images, valid_labels = sess.run([self.test_image_batch, self.test_image_batch])
            
            for i in range(5):
                img = Image.fromarray(np.asarray(batch_images[i]))
                display(img)
                print ("Label:", batch_labels[i], "- Marked as", "GOOD" if np.argmax(batch_labels[i]) == 0 else "BAD")
        
            coord.request_stop()
            coord.join(threads)
            sess.close()
        
        
    def init_weights(self, shape, layer_name):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=layer_name)
    
    # MODEL
    def model(self, X, W, b, p_keep_conv, p_keep_hidden):
        with tf.name_scope("Conv1") as scope:
            if (self.debug):
                tf.summary.histogram("Weight_Conv1", W['wc1'])
                tf.summary.histogram("Bias_Conv1", b['bc1'])

            conv1 = tf.nn.conv2d(X, W['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            conv1_a = tf.nn.relu(conv1) + b['bc1']
            conv1 = tf.nn.max_pool(conv1_a, ksize=[1, self.max_pool_lost, self.max_pool_lost, 1], \
                                   strides=[1, self.max_pool_lost, self.max_pool_lost, 1], padding='SAME')
            conv1 = tf.nn.dropout(conv1, p_keep_conv)

        with tf.name_scope("Conv2") as scope:
            if (self.debug):
                tf.summary.histogram("Weight_Conv2", W['wc2'])
                tf.summary.histogram("Bias_Conv2", b['bc2'])

            conv2 = tf.nn.conv2d(conv1, W['wc2'], strides=[1, 1, 1, 1], padding='SAME')
            conv2_a = tf.nn.relu(conv2) + b['bc2']
            conv2 = tf.nn.max_pool(conv2_a, ksize=[1, self.max_pool_lost, self.max_pool_lost, 1], \
                                   strides=[1, self.max_pool_lost, self.max_pool_lost, 1], padding='SAME')
            conv2 = tf.nn.dropout(conv2, p_keep_conv)

        with tf.name_scope("Conv3") as scope:
            if (self.debug):
                tf.summary.histogram("Weight_Conv3", W['wc3'])
                tf.summary.histogram("Bias_Conv3", b['bc3'])

            conv3 = tf.nn.conv2d(conv2, W['wc3'], strides=[1, 1, 1, 1], padding='SAME')
            conv3_a = tf.nn.relu(conv3) + b['bc3']
            conv3 = tf.nn.max_pool(conv3, ksize=[1, self.max_pool_lost, self.max_pool_lost, 1], \
                                      strides=[1, self.max_pool_lost, self.max_pool_lost, 1], padding='SAME')
            conv3 = tf.nn.dropout(conv3, p_keep_conv)

        with tf.name_scope("FC_layer") as scope:
            if (self.debug):
                tf.summary.histogram("Weight_FC", W['wfc'])
                tf.summary.histogram("Bias_FC", b['bfc'])

            FC_layer = tf.reshape(conv3, [-1, W['wfc'].get_shape().as_list()[0]])
            FC_layer = tf.nn.relu(tf.matmul(FC_layer, W['wfc']))
            FC_layer = tf.nn.dropout(FC_layer, p_keep_hidden)

        with tf.name_scope("Softmax") as scope:
            if (self.debug):
                tf.summary.histogram("Weight_Softmax", W['wsm'])
                tf.summary.histogram("Bias_Softmax", b['bsm'])

    #         output_layer = tf.add(tf.matmul(FC_layer, W['wsm']), b['bsm'])

        output_layer = tf.add(tf.matmul(FC_layer, W['wsm']), b['bsm'], name="output")

        return output_layer
    
    def file_hash(self, filename):
        h = hashlib.sha256()
        with open(filename, 'rb', buffering=0) as f:
            for b in iter(lambda : f.read(128*1024), b''):
                h.update(b)
        return h.hexdigest()