# Machine Learning CS 6375 Project
# On Deep Dark Light.

# Team members:
# Swapnil Bansal sxb180020
# Harshel Jain hxj170009
# Vyom Gupta fxv180000
# Lipsa Senapati lxs180002

from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np
import rawpy
import glob
import PIL as pillow
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def upsample_merge(conv_out_1, conv_out_2, output_channels, input_channels):
    pool_size = 2
    deconvolution_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, input_channels], stddev=0.02))
    deconvolution = tf.nn.conv2d_transpose(conv_out_1, deconvolution_filter, tf.shape(conv_out_2), strides=[1, pool_size, pool_size, 1])
    deconvolution_output = tf.concat([deconvolution, conv_out_2], 3)
    deconvolution_output.set_shape([None, None, None, output_channels * 2])
    return deconvolution_output

def u_net(input):
    c1 = layer.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv1_1')
    c1 = layer.conv2d(c1, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv1_2')
    p1 = layer.max_pool2d(c1, [2, 2], padding='SAME')

    c2 = layer.conv2d(p1, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv2_1')
    c2 = layer.conv2d(c2, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv2_2')
    p2 = layer.max_pool2d(c2, [2, 2], padding='SAME')

    c3 = layer.conv2d(p2, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv3_1')
    c3 = layer.conv2d(c3, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv3_2')
    p3 = layer.max_pool2d(c3, [2, 2], padding='SAME')

    c4 = layer.conv2d(p3, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv4_1')
    c4 = layer.conv2d(c4, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv4_2')
    p4 = layer.max_pool2d(c4, [2, 2], padding='SAME')

    c5 = layer.conv2d(p4, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv5_1')
    c5 = layer.conv2d(c5, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv5_2')

    up6 = upsample_merge(c5, c4, 256, 512)
    c6 = layer.conv2d(up6, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv6_1')
    c6 = layer.conv2d(c6, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv6_2')

    up7 = upsample_merge(c6, c3, 128, 256)
    c7 = layer.conv2d(up7, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv7_1')
    c7 = layer.conv2d(c7, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv7_2')

    up8 = upsample_merge(c7, c2, 64, 128)
    c8 = layer.conv2d(up8, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv8_1')
    c8 = layer.conv2d(c8, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv8_2')

    up9 = upsample_merge(c8, c1, 32, 64)
    c9 = layer.conv2d(up9, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv9_1')
    c9 = layer.conv2d(c9, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv9_2')

    c10 = layer.conv2d(c9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out_image = tf.depth_to_space(c10, 2)
    return out_image

# pack Bayer image to 4 channels
def pack_raw(raw):
   
    # converts rawpy imreaded input raw using rawpy 'raw_image_visible' which removes from 
    # raw image without the dark borders of the sensor (usually used for noise measurements) 
    # to 32 bit data.
    image = raw.raw_image_visible.astype(np.float32)

    # subtract the black level, 14-bit data = 2^14. *Guess is 512 is nearest power of 2 to
    # mean of black level in dark frame for sensor.
    image = np.maximum(image - 2048, 0) / (16383 - 2048)

    #reshape image from HxW to add axis to end, HxWxnew_axis, i.e get data at 5,6: image[5,6,0]
    image = np.expand_dims(image, axis=2)

    # get height and width of image
    image_shape = image.shape
    height = image_shape[0]
    width = image_shape[1]
    
    # put 4 bayer channels in axis 2, i.e. out.shape = HxWx4
    output = np.concatenate((image[0:height:2, 0:width:2, :], image[0:height:2, 1:width:2, :], image[1:height:2, 1:width:2, :], image[1:height:2, 0:width:2, :]), axis=2)
    return output


if __name__ == "__main__":

    input_images = './Canon/short/'
    groundTruth_images = './Canon/long/'
    checkpoint_directory = './checkpoint/Canon/'
    output_Canon = './output_Canon/'
    groundTruth_Canon = './groundTruth_Canon/'
    scaled_Canon = './scaled_Canon/'
    dark_Canon = './dark_Canon/'

    if not os.path.isdir(output_Canon ):
        os.makedirs(output_Canon)

    if not os.path.isdir(groundTruth_Canon):
        os.makedirs(groundTruth_Canon)

    if not os.path.isdir(scaled_Canon):
        os.makedirs(scaled_Canon )

    if not os.path.isdir(dark_Canon ):
        os.makedirs(dark_Canon)

    # fetching test data
    testArr = glob.glob(groundTruth_images + '/1*.CR2')
    prefixes = [int(os.path.basename(arr)[0:5]) for arr in testArr]

    session = tf.Session()
    input_image = tf.placeholder(tf.float32, [None, None, None, 4])
    ground_truth_image = tf.placeholder(tf.float32, [None, None, None, 3])
    output_image = u_net(input_image)

    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    
    if ckpt:
        print('Loaded',ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)

    for prefix in prefixes:
        # test the first image in each sequence
        input_files = glob.glob(input_images + '%05d_00*.CR2' % prefix)
        for k in range(len(input_files)):
            
            file_name = os.path.basename(input_files[k])
            print("Processing file:",file_name)

            ground_truth_file = glob.glob(groundTruth_images + '%05d_00*.CR2' % prefix)
            ground_truth_filename = os.path.basename(ground_truth_file[0])

            input_exposure = float(file_name[9:-5])
            groundTruth_exposure = float(ground_truth_filename[9:-5])
            ratio = min(groundTruth_exposure / input_exposure, 300)

            raw = rawpy.imread(input_files[k])
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # scale_full = np.expand_dims(np.float32(image/65535.0),axis = 0)*ratio
            scale_full = np.expand_dims(np.float32(image / 65535.0), axis=0)
            
            # output dark image
            dark = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, no_auto_scale=True, output_bps=16)

            groundTruth_raw = rawpy.imread(ground_truth_file[0])
            image = groundTruth_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            groundTruth_full = np.expand_dims(np.float32(image / 65535.0), axis=0)


            input_full = np.minimum(input_full, 1.0)

            output = session.run(output_image, feed_dict={input_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            groundTruth_full = groundTruth_full[0, :, :, :]
            scale_full = scale_full[0, :, :, :]
            scale_full = scale_full * np.mean(groundTruth_full) / np.mean(scale_full)  # scale the low-light image to the same mean of the groundtruth

            Image.fromarray((dark).astype(np.uint8)).save(dark_Canon + '%5d_00_%d.png' % (prefix, ratio))
            Image.fromarray((output * 256).astype(np.uint8)).save(output_Canon + '%5d_00_%d.png' % (prefix, ratio))
            Image.fromarray((scale_full * 256).astype(np.uint8)).save(scaled_Canon + '%5d_00_%d.png' % (prefix, ratio))
            Image.fromarray((groundTruth_full * 256).astype(np.uint8)).save(groundTruth_Canon + '%5d_00_%d.png' % (prefix, ratio))
