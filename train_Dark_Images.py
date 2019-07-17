# Machine Learning CS 6375 Project
# On Deep Dark Light.

# Team members:
# Swapnil Bansal sxb180020
# Harshel Jain hxj170009
# Vyom Gupta fxv180000
# Lipsa Senapati lxs180002

from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

patch_size = 512  # patch size for training
save_frequency = 500

DEBUG = 0
if DEBUG == 1:
    save_frequency = 2
    prefixes = prefixes[0:5]

def upsample_merge(conv_out_1, conv_out_2, output_channels, input_channels):
    pool_size = 2
    deconvolution_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, input_channels], stddev=0.02))
    deconvolution = tf.nn.conv2d_transpose(conv_out_1, deconvolution_filter, tf.shape(conv_out_2), strides=[1, pool_size, pool_size, 1])
    deconvolution_output = tf.concat([deconvolution, conv_out_2], 3)
    deconvolution_output.set_shape([None, None, None, output_channels * 2])
    return deconvolution_output

def u_net(input):
    c1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv1_1')
    c1 = slim.conv2d(c1, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv1_2')
    p1 = slim.max_pool2d(c1, [2, 2], padding='SAME')

    c2 = slim.conv2d(p1, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv2_1')
    c2 = slim.conv2d(c2, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv2_2')
    p2 = slim.max_pool2d(c2, [2, 2], padding='SAME')

    c3 = slim.conv2d(p2, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv3_1')
    c3 = slim.conv2d(c3, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv3_2')
    p3 = slim.max_pool2d(c3, [2, 2], padding='SAME')

    c4 = slim.conv2d(p3, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv4_1')
    c4 = slim.conv2d(c4, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv4_2')
    p4 = slim.max_pool2d(c4, [2, 2], padding='SAME')

    c5 = slim.conv2d(p4, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv5_1')
    c5 = slim.conv2d(c5, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv5_2')

    up6 = upsample_merge(c5, c4, 256, 512)
    c6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv6_1')
    c6 = slim.conv2d(c6, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv6_2')

    up7 = upsample_merge(c6, c3, 128, 256)
    c7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv7_1')
    c7 = slim.conv2d(c7, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv7_2')

    up8 = upsample_merge(c7, c2, 64, 128)
    c8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv8_1')
    c8 = slim.conv2d(c8, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv8_2')

    up9 = upsample_merge(c8, c1, 32, 64)
    c9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv9_1')
    c9 = slim.conv2d(c9, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, scope='g_conv9_2')

    c10 = slim.conv2d(c9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
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

    # get train IDs
    trainArr = glob.glob(groundTruth_images + '0*.CR2')
    prefixes = [int(os.path.basename(arr)[0:5]) for arr in trainArr]

    #print(prefixes)
    #print('\n\n\n')

    session = tf.Session()
    input_image = tf.placeholder(tf.float32, [None, None, None, 4])
    ground_truth_image = tf.placeholder(tf.float32, [None, None, None, 3])
    output_image = u_net(input_image)

    total_loss = tf.reduce_mean(tf.abs(output_image - ground_truth_image))

    train_variables = tf.trainable_variables()
    learningRate = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(total_loss)

    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    if ckpt:
        print('Loaded ' + ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)

    # Raw data takes long time to load. Keep them in memory after loaded.
    groundTruth_images_list = [None] * 6000
    input_images_dict = {}

    #get gain ratios from files and create keys in input_images_dict

    for index in range(0, len(prefixes)):
            input_files = glob.glob(input_images + '%05d_00*.CR2' % prefixes[index])
            
            # debug
            #print(index)
            #print(prefixes[index])
            #print(input_files)

            input_file = input_files[0]
            file_name = os.path.basename(input_file)

            ground_truth_files = glob.glob(groundTruth_images + '%05d_00*.CR2' % prefixes[index])
            ground_truth_filename = os.path.basename(ground_truth_files[0])
            
            input_exposure = float(file_name[9:-5])
            groundTruth_exposure = float(ground_truth_filename[9:-5])
            ratio = int(min(groundTruth_exposure / input_exposure, 300))

            input_images_dict[str(ratio)] = [None] * len(prefixes)

    print('\ninput image keys:\n')
    print(input_images_dict.keys())
    print('\n')

    loss = np.zeros((5000, 1))

    all_folders = glob.glob('./result/*0')
    last_epoch = 0
    for folder in all_folders:
        last_epoch = np.maximum(last_epoch, int(folder[-4:]))

    learning_rate = 1e-4
    for epoch in range(last_epoch, 4001): # 4001
        if os.path.isdir("result/%04d" % epoch):
            continue
        count = 0
        if epoch > 2000:
            learning_rate = 1e-5

        for index in np.random.permutation(len(prefixes)):
            # get the path from image id
            prefixes[index] = prefixes[index]
            input_files = glob.glob(input_images + '%05d_00*.CR2' % prefixes[index])
            input_file = input_files[np.random.random_integers(0, len(input_files) - 1)]
            file_name = os.path.basename(input_file)

            ground_truth_files = glob.glob(groundTruth_images + '%05d_00*.CR2' % prefixes[index])
            ground_truth_files[0] = ground_truth_files[0]
            ground_truth_filename = os.path.basename(ground_truth_files[0])
            input_exposure = float(file_name[9:-5])
            groundTruth_exposure = float(ground_truth_filename[9:-5])
            ratio = int(min(groundTruth_exposure / input_exposure, 300))

            start_time = time.time()
            count += 1

            if input_images_dict[str(ratio)[0:3]][index] is None:
                raw = rawpy.imread(input_file)
                input_images_dict[str(ratio)[0:3]][index] = np.expand_dims(pack_raw(raw), axis=0) * ratio

                groundTruth_raw = rawpy.imread(ground_truth_files[0])
                image = groundTruth_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                groundTruth_images_list[index] = np.expand_dims(np.float32(image / 65535.0), axis=0)

            # crop
            height = input_images_dict[str(ratio)[0:3]][index].shape[1]
            width = input_images_dict[str(ratio)[0:3]][index].shape[2]

            x_array = np.random.randint(0, width - patch_size)
            y_array = np.random.randint(0, height - patch_size)

            input_patch = input_images_dict[str(ratio)[0:3]][index][:, y_array:y_array + patch_size, x_array:x_array + patch_size, :]
            groundTruth_patch = groundTruth_images_list[index][:, y_array * 2:y_array * 2 + patch_size * 2, x_array * 2:x_array * 2 + patch_size * 2, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_patch = np.flip(input_patch, axis=1)
                groundTruth_patch = np.flip(groundTruth_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=0)
                groundTruth_patch = np.flip(groundTruth_patch, axis=0)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                groundTruth_patch = np.transpose(groundTruth_patch, (0, 2, 1, 3))

            input_patch = np.minimum(input_patch, 1.0)

            _, loss[index], temp = session.run([optimizer, total_loss, output_image],
                                            feed_dict={input_image: input_patch, ground_truth_image: groundTruth_patch, learningRate: learning_rate})
            output = np.minimum(np.maximum(temp, 0), 1)

            print("%d %d Loss=%.3f Time=%.3f" % (epoch, count, np.mean(loss[np.where(loss)]), time.time() - start_time))

            if epoch % save_frequency == 0:
                if not os.path.isdir(output_Canon + '%04d' % epoch):
                    os.makedirs(output_Canon + '%04d' % epoch)

                result_image = np.concatenate((groundTruth_patch[0, :, :, :], output[0, :, :, :]), axis=1)
                Image.fromarray((result_image * 256).astype(np.uint8)).save(
                    output_Canon + '%04d/%05d_00_train_%d.jpg' % (epoch, prefixes[index], ratio))

        saver.save(session, checkpoint_directory + 'model.ckpt')
