from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def instancenorm(input):
    with tf.variable_scope("instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        #[batchsize ,1,1, channelNb]
        variance_epsilon = 1e-5
        #Batch normalization function does the mean substraction then divide by the standard deviation (to normalize it). It finally multiply by scale and adds offset.
        #normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        #For instanceNorm we do it ourselves :
        normalized = (((input - mean) / tf.sqrt(variance + variance_epsilon)) * scale) + offset
        return normalized, mean, variance


def deconv(batch_input, out_channels):
   with tf.variable_scope("deconv"):
        in_height, in_width, in_channels = [int(batch_input.get_shape()[1]), int(batch_input.get_shape()[2]), int(batch_input.get_shape()[3])]
        #filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter1 = tf.get_variable("filter1", [4, 4, out_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))

        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        resized_images = tf.image.resize_images(batch_input, [in_height * 2, in_width * 2], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv = tf.nn.conv2d(resized_images, filter, [1, 1, 1, 1], padding="SAME")
        conv = tf.nn.conv2d(conv, filter1, [1, 1, 1, 1], padding="SAME")

        #conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def fullyConnected(input, outputDim, useBias, layerName = "layer", initMultiplyer = 1.0):
    with tf.variable_scope("fully_connected"):
        batchSize = tf.shape(input)[0];
        inputChannels = int(input.get_shape()[-1])
        weights = tf.get_variable("weight", [inputChannels, outputDim ], dtype=tf.float32, initializer=tf.random_normal_initializer(0, initMultiplyer * tf.sqrt(1.0/float(inputChannels))))
        weightsTiled = tf.tile(tf.expand_dims(weights, axis = 0), [batchSize, 1,1])
        squeezedInput = input
        
        if (len(input.get_shape()) > 3) :
            squeezedInput = tf.squeeze(squeezedInput, [1])
            squeezedInput = tf.squeeze(squeezedInput, [1])
        #weightsTiled = tf.Print(weightsTiled,[tf.shape(weightsTiled)], "weightsTiled_dyn2 : ", summarize=10)          
        outputs = tf.matmul(tf.expand_dims(squeezedInput, axis = 1), weightsTiled)
        outputs = tf.squeeze(outputs, [1])
        if(useBias):
            bias = tf.get_variable("bias", [outputDim], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.002))
            outputs = outputs + tf.expand_dims(bias, axis = 0)
            
        return outputs


def GlobalToGenerator(inputs, channels):
    with tf.variable_scope("GlobalToGenerator1"):
        fc1 = fullyConnected(inputs, channels, False, "fullyConnected_global_to_unet" ,0.01)
    return tf.expand_dims(tf.expand_dims(fc1, axis = 1), axis=1)