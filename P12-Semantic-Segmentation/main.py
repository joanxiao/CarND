import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import glob
import shutil
import time

NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)
EPOCHS = 30
BATCH_SIZE = 16
KEEP_PROB = 0.7
LEARNING_RATE = 0.0001

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """   
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
   
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    return (tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name), 
           tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name), 
           tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name), 
           tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name), 
           tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name))
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """           
    #activation=tf.nn.relu,
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), name='layer7_1x1')
    layer7_upsampled = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), name='layer7_upsampled')
    
    #print('vgg_layer4_out: {}'.format(vgg_layer4_out.get_shape()))
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), name='layer4_1x1')    
    layer4_fuse = tf.add(layer7_upsampled, layer4_1x1)    
    layer4_upsampled = tf.layers.conv2d_transpose(layer4_fuse, num_classes, 4, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), name='layer4_upsampled')    

        
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), name='layer3_1x1')  
    layer3_fuse = tf.add(layer4_upsampled, layer3_1x1)        
       
    output = tf.layers.conv2d_transpose(layer3_fuse, num_classes, 16, strides=(8, 8), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), name='output')   
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')    
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    tf.summary.scalar('loss', cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
    
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, accuracy_op, iou, iou_op, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    train_writer = tf.summary.FileWriter('./tflog/train', tf.get_default_graph())      
    val_writer = tf.summary.FileWriter('./tflog/val', tf.get_default_graph())  
    saver = tf.train.Saver()
    
    step = 0
    start = time.time()
    print('training starts')
    for epoch_i in range(epochs):
                
        # The training cycle
        batch_i = 0
        for batch_features, batch_labels in get_batches_fn(batch_size, True):                      
            merge = tf.summary.merge_all()
            # log validation summary every 10 steps
            if step % 10 == 0: 
            	''' 
                # record validation loss. Commented out for final model training
                for val_features, val_labels in get_batches_fn(batch_size, False):                             
                    summary, l = sess.run(
                        [merge, cross_entropy_loss],
                        feed_dict={input_image: val_features, correct_label: val_labels, keep_prob:1.0})
                        
                    sess.run(iou_op, feed_dict={input_image: val_features, correct_label: val_labels, keep_prob:1.0})
                    mean_iou = sess.run(iou)
                    val_writer.add_summary(summary, step)
                    print('\nval loss:{}, mean_iou:{}'.format(l, mean_iou))      
                    '''               
            else:
                summary, _, l = sess.run(
                    [merge, train_op, cross_entropy_loss],
                    feed_dict={input_image: batch_features, correct_label: batch_labels, keep_prob:KEEP_PROB, learning_rate:LEARNING_RATE})
                                
                
                sess.run(iou_op, feed_dict={input_image: batch_features, correct_label: batch_labels, keep_prob:1.0})
                mean_iou = sess.run(iou)
                print('\nepoch:{}, batch:{}, loss:{}, mean_iou:{}'.format(epoch_i, batch_i, l, mean_iou))                                            
                train_writer.add_summary(summary, step)   
            batch_i += 1
            step += 1               
		
        if epoch_i > 20 and epoch_i % 2 == 0:
            saver.save(sess, 'model/kitti_model_{}'.format(epoch_i))
            print('model/kitti_model_{} saved'.format(epoch_i))
		        
    train_writer.close()
    val_writer.close()
    print('training finished. Time taken: {} minutes'.format((time.time() - start) / 60))
#tests.test_train_nn(train_nn)

def setup_train_val(root):   
    # copy images in training/ to new folder training_all/ (used for final model training after hyperparameter selection)
    # split images to train and val, and move val images to the folder val/
    # ground truth still remain in the original training/ folder
    train_dir = os.path.join(root, 'training/image_2')
    
    files = glob.glob(os.path.join(train_dir, '*.png'))
    train, val = train_test_split(files, test_size=0.3)  
    os.mkdir(os.path.join(root, 'val'))
    os.mkdir(os.path.join(root, 'val/image_2'))   
    
    shutil.copytree(train_dir, os.path.join(root, 'training_all/image_2'), ignore=shutil.ignore_patterns('calib', 'gt_image_2'))  
        
    for file in val:
        basename = os.path.basename(file)
        shutil.move(file, os.path.join(root, 'val/image_2', basename))       

# used to test various models        
def run_inference():	
    num_classes = NUM_CLASSES
    data_dir = './data'
    runs_dir = './runs'
    input_image = tf.placeholder(tf.float32, name='input_image')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    start = time.time()
    with tf.Session() as sess:          
        saver = tf.train.import_meta_graph('model/kitti_model_62.meta')
        saver.restore(sess,tf.train.latest_checkpoint('model/'))        
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
                
        graph = sess.graph        
        output = graph.get_tensor_by_name('logits:0')
        input_image = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob_1:0')
    
        logits = tf.reshape(output, (-1, num_classes)) 
        helper.save_inference_samples(runs_dir, data_dir, sess, IMAGE_SHAPE, logits, keep_prob, input_image)
        
    print('inference finished. Time taken: {} minutes'.format((time.time() - start) / 60))
    
def run():
    num_classes = NUM_CLASSES  
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    if not os.path.exists(os.path.join(data_dir, 'data_road', 'val')):
        print('setting up train/val sets')      
        setup_train_val(os.path.join(data_dir, 'data_road'))    
    
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
    
    
    with tf.Session() as sess:    
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        labels_reshaped = tf.reshape(correct_label, (-1, num_classes)) 
        logits, train_op, cross_entropy_loss = optimize(layers_output, labels_reshaped, learning_rate, num_classes)   
        
        #pred = tf.argmax(logits, 1)
        pred = tf.round(tf.nn.softmax(logits))       
        iou, iou_op = tf.metrics.mean_iou(labels_reshaped, pred, num_classes)
        tf.summary.scalar('iou', iou)
        
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_reshaped, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))       
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
                     
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, accuracy_op, iou, iou_op, input_image,
             correct_label, keep_prob, learning_rate)
                           
        helper.save_inference_samples(runs_dir, data_dir, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
    #run_inference()
  
