from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
from lxml import etree
from random import shuffle
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()

from models.input import inputMaterial
from constants.main import EPS, CROP_SIZE
from utils.parser import parse_arguments
from models.cnn import *

#Under MIT License
#Data was generated using Substance Designer and Substance Share library : https://share.allegorithmic.com. The data have its own license

#Source code tested for tensorflow version 1.4.0
# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64

args = parse_arguments()

if args.testMode == "auto":
    if args.input_dir.lower().endswith(".xml"):
        args.testMode = "xml";
    elif os.path.isdir(args.input_dir):
        args.testMode = "folder";
    else:
        args.testMode = "image";

Examples = collections.namedtuple("Examples", "iterator, paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, gen_loss_L1, gen_grads_and_vars, train, rerendered, gen_loss_L1_exact")

if args.depthFactor == 0:
    args.depthFactor = args.nbTargets

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2


def int_shape(x):
    return list(map(int, x.get_shape()))

def tf_generate_normalized_random_direction(batchSize, lowEps = 0.001, highEps =0.05):
    r1 = tf.random_uniform([batchSize, 1], 0.0 + lowEps, 1.0 - highEps, dtype=tf.float32)
    r2 =  tf.random_uniform([batchSize, 1], 0.0, 1.0, dtype=tf.float32)
    r = tf.sqrt(r1)
    phi = 2 * math.pi * r2
       
    x = r * tf.cos(phi)
    y = r * tf.sin(phi)
    z = tf.sqrt(1.0 - tf.square(r))
    finalVec = tf.concat([x, y, z], axis=-1) #Dimension here is [batchSize, 3]
    return finalVec
    
def tf_generate_distance(batchSize):
    gaussian = tf.random_normal([batchSize, 1], 0.5, 0.75, dtype=tf.float32) # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (tf.exp(gaussian))
    
def createMaterialTable(examplesDict, shuffleImages):
    materialsList = []
    pathsList = []
    flatPathsList = []
    examplesDictKeys = examplesDict.keys()
    examplesDictKeys = sorted(examplesDict)
    
    for substanceName in examplesDictKeys:
        #print(substanceName)
        for variationKey, variationList in examplesDict[substanceName].items():
            materialsList.append(variationList)
            tmpPathList = []
            if args.mode == "test":
                for variation in variationList:
                    tmpPathList.append(variation.path)
            else:
                if len(variationList) > 1:
                    randomChoices = np.random.choice(variationList, 2, replace = False)
                    tmpPathList.append(randomChoices[0].path)
                    tmpPathList.append(randomChoices[1].path)
                else:
                    tmpPathList.append(variationList[0].path)
            pathsList.append(tmpPathList)
    if shuffleImages == True:
        shuffle(pathsList)

    for elem in pathsList:
        #if len(elem) < 2:
        #    print(elem)
        flatPathsList.extend(elem)
    return flatPathsList
    #Here we have a list of smallVariations 

def readInputImage(inputPath):
    return [inputPath]
        
    
def readInputFolder(input_dir, shuffleList):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
        
    pathList = glob.glob(os.path.join(input_dir, "*." + args.imageFormat))
    pathList = sorted(pathList);
    
    if shuffleList:
        shuffle(pathList)
    return pathList
    
def readInputXML(inputPath, shuffleList):
    exampleDict = {}
    pathDict = {}
    tree = etree.parse(inputPath)
    for elem in tree.findall('.//item'):
        imagePath = elem.find('image').text
        if not (imagePath is None) and os.path.exists(imagePath):
            lightPower = elem.find('lightPower').text
            lightXPos = elem.find('lightXPos').text
            lightYPos = elem.find('lightYPos').text
            lightZPos = elem.find('lightZPos').text
            camXPos = elem.find('camXPos').text
            camYPos = elem.find('camYPos').text
            camZPos = elem.find('camZPos').text
            uvscale = elem.find('uvscale').text
            uoffset = elem.find('uoffset').text
            voffset = elem.find('voffset').text
            rotation = elem.find('rotation').text
            identifier = elem.find('identifier').text
            
            substanceName = imagePath.split("/")[-1]
            if(substanceName.split('.')[0].isdigit()):
                substanceName = '%04d' % int(substanceName.split('.')[0])
            substanceNumber = 0
            imageSplitsemi = imagePath.split(";")
            if len(imageSplitsemi) > 1:                    
                substanceName = imageSplitsemi[1]
                substanceNumber = imageSplitsemi[2].split(".")[0]
            #def __init__(self, name, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, path):

            material = inputMaterial(substanceName, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, imagePath)
            idkey = str(substanceNumber) +";"+ identifier.rsplit(";", 1)[0]
            
            if not (substanceName in exampleDict) :
                exampleDict[substanceName] = {idkey : [material]}
                pathDict[imagePath] = material # Add only a path to be queried as for each image we will grab the others that are alike with the other dict

            else:
                if not (idkey in exampleDict[substanceName]):
                    exampleDict[substanceName][idkey] = [material]
                    pathDict[imagePath] = material # Add only a path to be queried as for each image we will grab the others that are alike with the other dict

                else:
                    exampleDict[substanceName][idkey].append(material)
    print("dict length : " + str(len(exampleDict.items())))
    flatPathList = createMaterialTable(exampleDict, shuffleList)
    return flatPathList

def _parse_function(filename):
    image_string = tf.read_file(filename)
    raw_input = tf.image.decode_image(image_string)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
    
    
    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])
        images=[]
        input = raw_input
        #add black images as targets (this is a hack and should be removed for a real test code pipeline).
        if args.mode == "eval":
            shape = tf.shape(input)
            black = tf.zeros([shape[0], shape[1]  * args.nbTargets, shape[2]], dtype=tf.float32)
            input = tf.concat([input, black], axis=1)
        width = tf.shape(input)[1] # [height, width, channels]
        imageWidth = width // (args.nbTargets + 1)

        for imageId in range(args.nbTargets + 1):
            beginning = imageId * imageWidth
            end = (imageId+1) * imageWidth
            images.append(input[:,beginning:end,:])

    if args.which_direction == "AtoB":
        inputs, targets = [images[0], images[1:]]
    elif args.which_direction == "BtoA":
        inputs, targets = [images[-1], images[:-1]]
    else:
        raise Exception("invalid direction")
        
    if args.correctGamma:
        inputs = tf.pow(inputs, 2.2)

    if args.useLog:
        inputs = logTensor(inputs)
    inputs = preprocess(inputs)
    targetsTmp = []
    for target in targets:
        targetsTmp.append(preprocess(target))
    targets = targetsTmp
    # synchronize seed for image operations so that we do the same operations to both
    # input and output images 
    
    def transform(image):
        r = image
        #if a.flip:
        #    r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = []
        for target in targets:
            target_images.append(transform(target))
    
    return filename, input_images, target_images    

def load_examples(input_dir, shuffleValue):
    test_queue = tf.constant([" "])
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    flatPathList = []
    if args.testMode == "xml":
        flatPathList = readInputXML(input_dir, shuffleValue)
    elif args.testMode == "folder":
        flatPathList = readInputFolder(input_dir, shuffleValue)
    elif args.testMode == "image":
        flatPathList = readInputImage(input_dir)
    
    if len(flatPathList) == 0:
        raise Exception("input_dir contains no image files")
    with tf.name_scope("load_images"):
        filenamesTensor = tf.constant(flatPathList) 
        dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)
        dataset = dataset.map(_parse_function, num_parallel_calls=1)
        dataset = dataset.repeat()
        batched_dataset = dataset.batch(args.batch_size)
        #batched_dataset = batched_dataset.filter(lambda paths, blah, _: tf.equal(tf.shape(paths)[0], a.batch_size))

        iterator = batched_dataset.make_initializable_iterator()
        paths_batch, inputs_batch, targets_batch = iterator.get_next()
        
        if args.scale_size > CROP_SIZE:
            xyCropping = tf.random_uniform([2], 0, args.scale_size - CROP_SIZE, dtype=tf.int32)
            inputs_batch = inputs_batch[:, xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1] : xyCropping[1] + CROP_SIZE, :]
            targets_batch = targets_batch[:,:, xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1] : xyCropping[1] + CROP_SIZE, :]
        #paths_batch.set_shape([a.batch_size])
        print("targets_batch_0 : " + str(targets_batch.get_shape()))        
        
        inputs_batch.set_shape([None, CROP_SIZE, CROP_SIZE, inputs_batch.get_shape()[-1] ])
        targets_batch.set_shape([None, args.nbTargets, CROP_SIZE, CROP_SIZE, targets_batch.get_shape()[-1] ])

        
    #paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size, num_threads=1)
    tf.summary.text("batch paths", paths_batch)
    
    steps_per_epoch = int(math.floor(len(flatPathList) / args.batch_size))
    print("steps per epoch : " + str(steps_per_epoch))
    #[batchsize, nbMaps, 256,256,3] Do the reshape by hand and probably concat it on third axis so we are sure of the reshape.
    print("inputs_batch : " + str(inputs_batch.get_shape()))    
    print("targets_batch : " + str(targets_batch.get_shape()))
    targets_batch_concat = targets_batch[:,0]
    print("targets_batch_concat : " + str(targets_batch_concat.get_shape()))
    for imageId in range(1, args.nbTargets):
        targets_batch_concat = tf.concat([targets_batch_concat, targets_batch[:,imageId]], axis = -1)
    
    targets_batch = targets_batch_concat
    print("targets size : " + str(targets_batch.get_shape()))

    return Examples(
        iterator=iterator,
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(flatPathList),
        steps_per_epoch=steps_per_epoch,
    )
    
def logTensor(tensor):
    return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))
    
def create_generator(generator_inputs, generator_outputs_channels):
    print("generator_inputs :" + str(generator_inputs.get_shape()))
    print("generator_outputs_channels :" + str(generator_outputs_channels))
    layers = []
    #Input here should be [batch, 256,256,3]
    inputMean, inputVariance = tf.nn.moments(generator_inputs, axes=[1, 2], keep_dims=False)
    globalNetworkInput = inputMean
    globalNetworkOutputs = []
    with tf.variable_scope("globalNetwork_fc_1"):    
        globalNetwork_fc_1 = fullyConnected(globalNetworkInput, args.ngf * 2, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
        globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc_1))
        
    #encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, args.ngf * args.depthFactor , stride=2)
        layers.append(output)
    #Default ngf is 64
    layer_specs = [
        args.ngf * 2 * args.depthFactor, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        args.ngf * 4 * args.depthFactor, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        args.ngf * 8 * args.depthFactor, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        args.ngf * 8 * args.depthFactor, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        args.ngf * 8 * args.depthFactor, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        args.ngf * 8 * args.depthFactor, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        #a.ngf * 8 * a.depthFactor, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    
    for layerCount, out_channels in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            #here mean and variance will be [batch, 1, 1, out_channels]
            outputs, mean, variance = instancenorm(convolved)
            
            outputs = outputs + GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
            with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):  
                nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)  
                globalNetwork_fc = ""
                if layerCount + 1 < len(layer_specs) - 1:
                    globalNetwork_fc = fullyConnected(nextGlobalInput, layer_specs[layerCount + 1], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                else : 
                    globalNetwork_fc = fullyConnected(nextGlobalInput, layer_specs[layerCount], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
    
                globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
            layers.append(outputs)

    with tf.variable_scope("encoder_8"):
        rectified = lrelu(layers[-1], 0.2)
         # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        convolved = conv(rectified, args.ngf * 8 * args.depthFactor, stride=2)
        convolved = convolved  + GlobalToGenerator(globalNetworkOutputs[-1], args.ngf * 8 * args.depthFactor)
        
        with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):  
            mean, variance = tf.nn.moments(convolved, axes=[1, 2], keep_dims=True)
            nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
            globalNetwork_fc = fullyConnected(nextGlobalInput, args.ngf * 8 * args.depthFactor, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
            globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))  
                      
        layers.append(convolved)
    #default nfg = 64
    layer_specs = [
        (args.ngf * 8 * args.depthFactor, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (args.ngf * 8 * args.depthFactor, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (args.ngf * 8 * args.depthFactor, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (args.ngf * 8 * args.depthFactor, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (args.ngf * 4 * args.depthFactor, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (args.ngf * 2 * args.depthFactor, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (args.ngf * args.depthFactor, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = lrelu(input, 0.2)
            
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output, mean, variance = instancenorm(output)
            output = output + GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
            with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):    
                nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                globalNetwork_fc = fullyConnected(nextGlobalInput, out_channels, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = lrelu(input, 0.2)
        output = deconv(rectified, generator_outputs_channels)
        output = output + GlobalToGenerator(globalNetworkOutputs[-1], generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]
    
def tf_generateDiffuseRendering(batchSize, targets, outputs):    
    currentViewPos = tf_generate_normalized_random_direction(batchSize)
    currentLightPos = tf_generate_normalized_random_direction(batchSize)
    
    wi = currentLightPos
    wi = tf.expand_dims(wi, axis=1)
    wi = tf.expand_dims(wi, axis=1)
    
    wo = currentViewPos
    wo = tf.expand_dims(wo, axis=1)
    wo = tf.expand_dims(wo, axis=1)
    #[result, D_rendered, G_rendered, F_rendered, diffuse_rendered, specular_rendered]
    renderedDiffuse = tf_Render(targets,wi,wo)   
    
    renderedDiffuseOutputs = tf_Render(outputs,wi,wo)#tf_Render_Optis(outputs,wi,wo)
    return [renderedDiffuse, renderedDiffuseOutputs]

def tf_generateSpecularRendering(batchSize, surfaceArray, targets, outputs):    

    currentViewDir = tf_generate_normalized_random_direction(batchSize)
    currentLightDir = currentViewDir * tf.expand_dims([-1.0, -1.0, 1.0], axis = 0)
    #Shift position to have highlight elsewhere than in the center.
    currentShift = tf.concat([tf.random_uniform([batchSize, 2], -1.0, 1.0), tf.zeros([batchSize, 1], dtype=tf.float32) + 0.0001], axis=-1)
    
    currentViewPos = tf.multiply(currentViewDir, tf_generate_distance(batchSize)) + currentShift
    currentLightPos = tf.multiply(currentLightDir, tf_generate_distance(batchSize)) + currentShift
    
    currentViewPos = tf.expand_dims(currentViewPos, axis=1)
    currentViewPos = tf.expand_dims(currentViewPos, axis=1)

    currentLightPos = tf.expand_dims(currentLightPos, axis=1)
    currentLightPos = tf.expand_dims(currentLightPos, axis=1)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    renderedSpecular = tf_Render(targets,wi,wo, includeDiffuse = args.includeDiffuse)           
    renderedSpecularOutputs = tf_Render(outputs,wi,wo, includeDiffuse = args.includeDiffuse)#tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)
    return [renderedSpecular, renderedSpecularOutputs]
    
def DX(x):
    return x[:,:,1:,:] - x[:,:,:-1,:]    # so this just subtracts the image from a single-pixel shifted version of itself (while cropping out two pixels because we don't know what's outside the image)

def DY(x):
    return x[:,1:,:,:] - x[:,:-1,:,:]    # likewise for y-direction

def loss_l1(x, y):
    return tf.reduce_mean(tf.abs(x-y))

def loss_l2(x, y):
    return tf.reduce_mean(tf.square(x-y))

def loss_grad(x, y, alpha=0.2):
    loss_val = alpha * loss_l1(x,y) # here alpha is a weighting for the direct pixel value comparison, you can make it surprisingly low, though it's possible that 0.1 might be too low for some problems
    loss_val = loss_val + loss_l1(DX(x), DX(y))
    loss_val = loss_val + loss_l1(DY(x), DY(y))
    return loss_val
    

def create_model(inputs, targets, reuse_bool = False):
    batchSize = tf.shape(inputs)[0]
    surfaceArray=[]
    if args.loss == "render" or args.loss == "renderL2":
        XsurfaceArray = tf.expand_dims(tf.lin_space(-1.0, 1.0, CROP_SIZE), axis=-1)
        XsurfaceArray = tf.tile(XsurfaceArray,[1,CROP_SIZE])
        YsurfaceArray = -1 * tf.transpose(XsurfaceArray) #put -1 in the bottom of the table
        XsurfaceArray = tf.expand_dims(XsurfaceArray, axis = -1)
        YsurfaceArray = tf.expand_dims(YsurfaceArray, axis = -1)

        surfaceArray = tf.concat([XsurfaceArray, YsurfaceArray, tf.zeros([CROP_SIZE, CROP_SIZE,1], dtype=tf.float32)], axis=-1)
        surfaceArray = tf.expand_dims(surfaceArray, axis = 0) #Add dimension to support batch size

    with tf.variable_scope("generator", reuse=reuse_bool) as scope:
        out_channels = 9
        outputs = create_generator(inputs, out_channels) 
        
    partialOutputedNormals = outputs[:,:,:,0:2]
    outputedDiffuse = outputs[:,:,:,2:5]
    outputedRoughness = outputs[:,:,:,5]
    outputedSpecular = outputs[:,:,:,6:9]
    normalShape = tf.shape(partialOutputedNormals)
    newShape = [normalShape[0], normalShape[1], normalShape[2], 1]

    tmpNormals = tf.ones(newShape, tf.float32)
    
    normNormals = tf_Normalize(tf.concat([partialOutputedNormals, tmpNormals], axis = -1))
    outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
    reconstructedOutputs =  tf.concat([normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedSpecular], axis=-1)

    with tf.name_scope("generator_loss"):
        gen_loss_L1 = 0
        rerenderedTargets = []
        rerenderedOutputs = []
        renderedDiffuseImages = []
        renderedDiffuseImagesOutputs = []
        renderedSpecularImages = []
        renderedSpecularImagesOutputs = []
        
        outputs = reconstructedOutputs
        
        if args.loss == "l1":
            epsilon = 0.001
            NormalL1 = tf.abs(targets[0,:,:,0:3] - outputs[0,:,:,0:3]) * args.normalLossFactor
            DiffuseL1 = tf.abs(tf.log(epsilon + deprocess(targets[0,:,:,3:6])) - tf.log(epsilon + deprocess(outputs[0,:,:,3:6]))) * args.diffuseLossFactor
            RoughnessL1 = tf.abs(targets[0,:,:,6:9] - outputs[0,:,:,6:9]) * args.roughnessLossFactor
            SpecularL1 = tf.abs(tf.log(epsilon + deprocess(targets[0,:,:,9:12])) - tf.log(epsilon + deprocess(outputs[0,:,:,9:12]))) * args.specularLossFactor
            
            gen_loss_L1 = tf.reduce_mean(NormalL1 + DiffuseL1 + SpecularL1 + RoughnessL1)
        elif args.loss == "l2":
            epsilon = 0.001
            NormalL1 = tf.square(targets[0,:,:,0:3] - outputs[0,:,:,0:3]) * args.normalLossFactor
            DiffuseL1 = tf.square(tf.log(epsilon + deprocess(targets[0,:,:,3:6])) - tf.log(epsilon + deprocess(outputs[0,:,:,3:6]))) * args.diffuseLossFactor
            RoughnessL1 = tf.square(targets[0,:,:,6:9] - outputs[0,:,:,6:9]) * args.roughnessLossFactor
            SpecularL1 = tf.square(tf.log(epsilon + deprocess(targets[0,:,:,9:12])) - tf.log(epsilon + deprocess(outputs[0,:,:,9:12]))) * args.specularLossFactor

            gen_loss_L1 = tf.reduce_mean(NormalL1 + DiffuseL1 + SpecularL1 + RoughnessL1)        
        elif args.loss == "render" or args.loss == "renderL2":
            with tf.name_scope("renderer"):
                with tf.name_scope("diffuse"):
                    for nbDiffuseRender in range(args.nbDiffuseRendering):
                        diffuses = tf_generateDiffuseRendering(batchSize, targets, outputs)
                        renderedDiffuseImages.append(diffuses[0][0])
                        renderedDiffuseImagesOutputs.append(diffuses[1][0]) 
                                                
                with tf.name_scope ("specular"):
                    for nbspecularRender in range(args.nbSpecularRendering):
                        speculars = tf_generateSpecularRendering(batchSize, surfaceArray, targets, outputs)
                        renderedSpecularImages.append(speculars[0][0])
                        renderedSpecularImagesOutputs.append(speculars[1][0]) 
                        
                        
                # renderedDiffuseImages contains X (3 by default) renderings of shape [batch_size, 256,256,3]
                rerenderedTargets = renderedDiffuseImages[0]
                for renderingDiff in renderedDiffuseImages[1:]:
                    rerenderedTargets = tf.concat([rerenderedTargets, renderingDiff], axis = -1)
                for renderingSpecu in renderedSpecularImages:
                    rerenderedTargets = tf.concat([rerenderedTargets, renderingSpecu], axis = -1)

                rerenderedOutputs = renderedDiffuseImagesOutputs[0]
                for renderingOutDiff in renderedDiffuseImagesOutputs[1:]:
                    rerenderedOutputs = tf.concat([rerenderedOutputs, renderingOutDiff], axis = -1)
                for renderingOutSpecu in renderedSpecularImagesOutputs:
                    rerenderedOutputs = tf.concat([rerenderedOutputs, renderingOutSpecu], axis = -1)               

                gen_loss_L1 = 0
                if args.loss == "render":
                    gen_loss_L1 = tf.reduce_mean(tf.abs(tf.log(rerenderedTargets + 0.01) - tf.log(rerenderedOutputs + 0.01)))
                if args.loss == "renderL2":
                    gen_loss_L1 = tf.reduce_mean(tf.square(tf.log(rerenderedTargets + 0.01) - tf.log(rerenderedOutputs + 0.01)))
        consistencyLoss = 0
                    
        gen_loss = gen_loss_L1 * args.l1_weight

    with tf.name_scope("generator_train"):
        with tf.variable_scope("generator_train0", reuse=reuse_bool):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss_L1, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([gen_loss_L1])
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    return Model(
        gen_loss_L1_exact=gen_loss_L1,
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
        rerendered = [rerenderedTargets,rerenderedOutputs]
    )
    
def save_loss_value(values):
    averaged = np.mean(values)
    with open(os.path.join(args.output_dir, "losses.txt"), "a") as f:
            f.write(str(averaged) + "\n")
                    
def save_images(fetches, output_dir = args.output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        #fetch inputs
        kind = "inputs"
        filename = name + "-" + kind + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][i]
        with open(out_path, "wb") as f:
            f.write(contents)
        #fetch outputs and targets
        for kind in ["outputs", "targets"]:
            for idImage in range(args.nbTargets):
                filename = name + "-" + kind + "-" + str(idImage) + "-.png"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
                filetsetKey = kind + str(idImage)
                fileset[filetsetKey] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i * args.nbTargets + idImage]
                with open(out_path, "wb") as f:
                    f.write(contents)
        filesets.append(fileset)

    return filesets


def append_index(filesets, output_dir = args.output_dir, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        mapnames = ["normals", "diffuse", "roughness", "log(specular)"]
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>log(input)</th>")
        for idImage in range(args.nbTargets):
            index.write("<th>" + str(mapnames[idImage]) + "</th>")
        index.write("</tr>")            

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s targets</td>" % fileset["name"])
        if args.mode != "eval" : 

            for kind in ["inputs", "targets"]:
                if kind == "inputs":
                    index.write("<td><img src='images/%s'></td>" % fileset[kind])
                elif kind == "targets":
                    for idImage in range(args.nbTargets):
                        filetsetKey = kind + str(idImage)
                        index.write("<td><img src='images/%s'></td>" % fileset[filetsetKey])
            index.write("</tr>")
            index.write("<tr>")

        if step:
            index.write("<td></td>")
        index.write("<td>outputs</td>")
        for kind in ["inputs", "outputs"]:
            if kind == "inputs":
                index.write("<td><img src='images/%s'></td>" % fileset[kind])
            elif kind=="outputs":
                for idImage in range(args.nbTargets):
                    filetsetKey = kind + str(idImage)
                    index.write("<td><img src='images/%s'></td>" % fileset[filetsetKey])
        index.write("</tr>")
    
    return index_path

def runTestFromTrain(currentStep, evalExamples, max_steps, display_fetches_test, sess):
    #sess.run(evalExamples.iterator.initializer)
    max_steps = min(evalExamples.steps_per_epoch, max_steps)
    for step in range(max_steps):
        try:
            results_test = sess.run(display_fetches_test)
            test_output_dir = args.output_dir + "/testStep"+str(currentStep)
            filesets = save_images(results_test, test_output_dir)
            index_path = append_index(filesets, test_output_dir)
        except tf.errors.OutOfRangeError:
            print("error in the runTestFromTrain of OutOfRangeError")
            continue;    
    print("wrote index at", index_path)
    
def reshape_tensor_display(tensor, splitAmount, logAlbedo = False):
    tensors_list = tf.split(tensor, splitAmount, axis=3)#4 * [batch, 256,256,3]
    if logAlbedo:
        tensors_list[-1] = logTensor(tensors_list[-1])
        tensors_list[1] = logTensor(tensors_list[1])
    
    tensors = tf.stack(tensors_list, axis = 1) #[batch, 4,256,256,3]
    shape = tf.shape(tensors)
    newShape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
    tensors_reshaped = tf.reshape(tensors, newShape)
    #print(tensors_reshaped.get_shape()) 
    return tensors_reshaped
    
def main():

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == "test" or args.mode == "export" or args.mode == "eval" :
        if args.checkpoint is None:
            raise Exception("checkpoint required for test, export or eval mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "nbTargets", "depthFactor", "loss", "useLog"}
        with open(os.path.join(args.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(args, key, val)
        # disable these features in test mode
        args.scale_size = CROP_SIZE
        args.flip = False

    for k, v in args._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    examples = load_examples(args.input_dir, args.mode == "train")
    print(args.mode + " set count = %d" % examples.count)
    if args.mode == "train":
        evalExamples = load_examples(args.input_dir.rsplit('\\',1)[0] + "\\testBlended", False)
        print("evaluation set count = %d" % evalExamples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets, False)
    if args.mode == "train":
        model_test = create_model(evalExamples.inputs, evalExamples.targets, True)

    tmpTargets = examples.targets
    if args.mode == "train":
        tmpTargetsTest = evalExamples.targets

    # undo colorization splitting on images that we use for display/output    
    inputs = deprocess(examples.inputs)
    targets = deprocess(tmpTargets)
    outputs = deprocess(model.outputs)
    
            
    if args.mode == "train":
        inputsTests = deprocess(evalExamples.inputs)
        targetsTests = deprocess(tmpTargetsTest)
        outputsTests = deprocess(model_test.outputs)

    def convert(image, squeeze=False):
        if args.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * args.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        if squeeze:
            def tempLog(imageValue):                    
                imageValue= tf.log(imageValue + 0.01)
                imageValue = imageValue - tf.reduce_min(imageValue)
                imageValue = imageValue / tf.reduce_max(imageValue)
                return imageValue
            image = [tempLog(imageVal) for imageVal in image]

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    with tf.name_scope("transform_images"):
        targets_reshaped = reshape_tensor_display(targets, args.nbTargets, logAlbedo = args.logOutputAlbedos)
        outputs_reshaped = reshape_tensor_display(outputs, args.nbTargets, logAlbedo = args.logOutputAlbedos)
        inputs_reshaped = reshape_tensor_display(inputs, 1, logAlbedo = False)

        if args.mode == "train":
            inputs_reshaped_test = reshape_tensor_display(inputsTests, 1, logAlbedo = False)
            targets_test_reshaped = reshape_tensor_display(targetsTests, args.nbTargets, logAlbedo = args.logOutputAlbedos)
            outputs_test_reshaped = reshape_tensor_display(outputsTests, args.nbTargets, logAlbedo = args.logOutputAlbedos)


    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs_reshaped)
        if args.mode == "train":
            converted_inputs_test = convert(inputs_reshaped_test)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets_reshaped)
        if args.mode == "train":
            converted_targets_test = convert(targets_test_reshaped)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs_reshaped)
        if args.mode == "train":
            converted_outputs_test = convert(outputs_test_reshaped)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        if args.mode == "train":
            display_fetches_test = {
                "paths": evalExamples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs_test, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets_test, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs_test, dtype=tf.string, name="output_pngs"),
            }        
            
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs, max_outputs=args.nbTargets)

    tf.summary.scalar("generator_loss", model.gen_loss_L1)


    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = args.output_dir if ( args.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))
        if args.checkpoint is not None:
            print("loading model from checkpoint : " + args.checkpoint)
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        
        sess.run(examples.iterator.initializer)
        print("BBBBBBBBbb")

        if args.mode == "test" or args.mode == "eval":
            print("AAAAAAAAAAAAAAA")

            if args.checkpoint is None:
                print("checkpoint is required for testing")
                return
            # testing
            # at most, process the test data once
            print("CCCCCCCCCCCCCCCCc")

            max_steps = min(examples.steps_per_epoch, max_steps)
            print(max_steps)
            for step in range(max_steps):
                try:
                    #display_fetches["rerenders"] = renders_fetches_test
                    #display_fetches["gen_loss_L1_exact"] = model.gen_loss_L1_exact
                    results = sess.run(display_fetches)
                    #save_tensor_images(results["rerenders"], "rerenderings/", suffix = step)
                    #L1Values.append(results["gen_loss_L1_exact"])
                    filesets = save_images(results)
                    for i, f in enumerate(filesets):
                        print("evaluated image", f["name"])
                    index_path = append_index(filesets)
                except tf.errors.OutOfRangeError :
                    print("testing fails in OutOfRangeError")
                    continue;
        else:
            try:
                # training
                start_time = time.time()
                sess.run(evalExamples.iterator.initializer)
                for step in range(max_steps):
                    def should(freq):
                        return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                    options = None
                    run_metadata = None

                    fetches = {
                        "train": model.train,
                        "global_step": sv.global_step,
                    }

                    if should(args.progress_freq) or step == 0 or step == 1:
                        fetches["gen_loss_L1"] = model.gen_loss_L1

                    if should(args.summary_freq):
                        fetches["summary"] = sv.summary_op

                    if should(args.display_freq):
                        fetches["display"] = display_fetches
                        
                    try:    
                        results = sess.run(fetches, options=options, run_metadata=run_metadata)
                    except tf.errors.OutOfRangeError :
                        print("training fails in OutOfRangeError")
                        continue
                        
                    global_step = results["global_step"]

                    if should(args.summary_freq):
                        sv.summary_writer.add_summary(results["summary"], global_step)

                    if should(args.display_freq):
                        print("saving display images")
                        filesets = save_images(results["display"], step=global_step)
                        append_index(filesets, step=True)

                    if should(args.progress_freq):
                        # global_step will have the correct step count if we resume from a checkpoint
                        train_epoch = math.ceil(global_step / examples.steps_per_epoch)
                        train_step = global_step - (train_epoch - 1) * examples.steps_per_epoch
                        print("progress  epoch %d  step %d  image/sec %0.1f" % (train_epoch, train_step, global_step * args.batch_size / (time.time() - start_time)))
                        print("gen_loss_L1", results["gen_loss_L1"])

                    if should(args.save_freq):
                        print("saving model")
                        saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)
                    if should(args.test_freq) or global_step == 1:
                        runTestFromTrain(global_step, evalExamples, max_steps, display_fetches_test, sess)
                        
                    if sv.should_stop():
                        break
            finally:
                saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)
                sess.run(evalExamples.iterator.initializer)
                runTestFromTrain("final", evalExamples, max_steps, display_fetches_test, sess)
                
                
# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keep_dims=True))
    return tf.div(tensor, Length)
    
    

# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keep_dims=True))
    return tf.div(tensor, Length)

# Computes the dot product between 2 tensors (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1). 
def tf_DotProduct(tensorA, tensorB):       
    return tf.reduce_sum(tf.multiply(tensorA, tensorB), axis = -1, keep_dims=True)

##########Rendering loss

def tf_render_diffuse_Substance(diffuse, specular):
    return diffuse * (1.0 - specular) / math.pi

def tf_render_D_GGX_Substance(roughness, NdotH):
    alpha = tf.square(roughness)
    underD = 1/tf.maximum(0.001, (tf.square(NdotH) * (tf.square(alpha) - 1.0) + 1.0))
    return (tf.square(alpha * underD)/math.pi)
    
def tf_lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT*tf.square(distance));


def tf_render_F_GGX_Substance(specular, VdotH):
    sphg = tf.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
    return specular + (1.0 - specular) * sphg
    
def tf_render_G_GGX_Substance(roughness, NdotL, NdotV):
    return G1_Substance(NdotL, tf.square(roughness)/2) * G1_Substance(NdotV, tf.square(roughness)/2)
    
def G1_Substance(NdotW, k):
    return 1.0/tf.maximum((NdotW * (1.0 - k) + k), 0.001)


def squeezeValues(tensor, min, max):
    return tf.clip_by_value(tensor, min, max)

# svbdrf : (BatchSize, Width, Height, 4 * 3)
# wo : (BatchSize,1,1,3)
# wi : (BatchSize,1,1,3) 
def tf_Render(svbrdf, wi, wo, includeDiffuse = True):
    wiNorm = tf_Normalize(wi)
    woNorm = tf_Normalize(wo)
    h = tf_Normalize(tf.add(wiNorm,woNorm) / 2.0)
    diffuse = squeezeValues(deprocess(svbrdf[:,:,:,3:6]), 0.0,1.0)
    normals = svbrdf[:,:,:,0:3]
    specular = squeezeValues(deprocess(svbrdf[:,:,:,9:12]), 0.0, 1.0)
    roughness = squeezeValues(deprocess(svbrdf[:,:,:,6:9]), 0.0, 1.0)
    roughness = tf.maximum(roughness, 0.001)
    NdotH = tf_DotProduct(normals, h)
    NdotL = tf_DotProduct(normals, wiNorm)
    NdotV = tf_DotProduct(normals, woNorm)
    VdotH = tf_DotProduct(woNorm, h)

    diffuse_rendered = tf_render_diffuse_Substance(diffuse, specular)
    D_rendered = tf_render_D_GGX_Substance(roughness, tf.maximum(0.0, NdotH))
    G_rendered = tf_render_G_GGX_Substance(roughness, tf.maximum(0.0, NdotL), tf.maximum(0.0, NdotV))
    F_rendered = tf_render_F_GGX_Substance(specular, tf.maximum(0.0, VdotH))
    
    
    specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
    result = specular_rendered
    
    if includeDiffuse:
        result = result + diffuse_rendered
    
    lampIntensity = 1.0
    #lampDistance = tf.sqrt(tf.reduce_sum(tf.square(wi), axis = 3, keep_dims=True))
    
    lampFactor = lampIntensity * math.pi#tf_lampAttenuation(lampDistance) * lampIntensity * math.pi
    
    result = result * lampFactor

    result = result * tf.maximum(0.0, NdotL) / tf.expand_dims(tf.maximum(wiNorm[:,:,:,2], 0.001), axis=-1) # This division is to compensate for the cosinus distribution of the intensity in the rendering

    return [result, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]
    

def main1():
    print(tf.config.list_physical_devices('GPU'))
    
main()
