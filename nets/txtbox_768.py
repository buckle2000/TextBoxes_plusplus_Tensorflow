# -*- coding: utf-8 -*-
"""
This framework is based on SSD_tensorlow(https://github.com/balancap/SSD-Tensorflow)
Add descriptions
"""

import math
from collections import namedtuple
import copy
import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import textbox_common

import tensorflow.contrib.slim as slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
TextboxParams = namedtuple('TextboxParameters',
                           ['img_shape',
                            'num_classes',
                            'feat_layers',
                            'feat_shapes',
                            'scale_range',
                            'anchor_ratios',
                            'anchor_sizes',
                            'anchor_steps',
                            'normalizations',
                            'prior_scaling',
                            'step',
                            'scales'
                            ])


class TextboxNet(object):
	"""
	Implementation of the Textbox 300 network.

	The default features layers with 300x300 image input are:
	  conv4_3 ==> 38 x 38
	  fc7 ==> 19 x 19
	  conv6_2 ==> 10 x 10
	  conv7_2 ==> 5 x 5
	  conv8_2 ==> 3 x 3
	  pool6 ==> 1 x 1
	The default image size used to train this network is 300x300.
	"""
	default_params = TextboxParams(
		img_shape=(768, 768),
		num_classes=2,
		feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11'],
		feat_shapes=[(96, 96), (48, 48), (24, 24), (12, 12), (10, 10), (8, 8)],
		scale_range=[0.20, 0.90],
		# anchor_ratios=[1.0, 2.0, 3.0,4.0, 5.0,1.0, 0.50, 1./3 , 1./4, 1./5],
		anchor_ratios=[
			[2.0, 1. / 2, 3.0, 1. / 3, 5.0, 1. / 5, 7.0, 1. / 7, 9.0, 1. / 9, 15.0, 1. /15],
			[2.0, 1. / 2, 3.0, 1. / 3, 5.0, 1. / 5, 7.0, 1. / 7, 9.0, 1. / 9, 15.0, 1. / 15],
			[2.0, 1. / 2, 3.0, 1. / 3, 5.0, 1. / 5, 7.0, 1. / 7, 9.0, 1. / 9, 15.0, 1. / 15],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5.0, 1. / 5, 7., 1. / 7, 9., 1. / 9],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5.0, 1. / 5, 7., 1. / 7, 9., 1. / 9],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5.0, 1. / 5],
		],
		anchor_sizes=[
			(30.,60.),
			(30.,90.),
			(90.,150.),
			(150., 210.),
			(210., 270.),
			(270., 330.)
		],
		anchor_steps=[8, 16, 32, 64, 100, 300],
		normalizations=[20, -1, -1, -1, -1, -1],
		prior_scaling=[0.1, 0.1, 0.2, 0.2],
		step=0.14,
		scales=[0.2, 0.34, 0.48, 0.62, 0.76, 0.90]
	)#step  8 16 32 64 96 192

	def __init__(self, params=None):
		"""
		Init the Textbox net with some parameters. Use the default ones
		if none provided.
		"""
		if isinstance(params, TextboxParams):
			self.params = params
		else:
			self.params = self.default_params
			# self.params.step = (scale_range[1] - scale_range[0])/ 5
			# self.params.scales = [scale_range[0] + i* self.params.step for i in range(6)]

	# ======================================================================= #
	def net(self, inputs,
	        is_training=True,
	        dropout_keep_prob=0.5,
	        reuse=None,
	        scope='text_box_384'):
		"""
		Text network definition.
		"""
		r = text_net(inputs,
		             feat_layers=self.params.feat_layers,
		             anchor_sizes=self.params.anchor_sizes,
		             anchor_ratios = self.params.anchor_ratios,
		             normalizations=self.params.normalizations,
		             is_training=is_training,
		             dropout_keep_prob=dropout_keep_prob,
		             reuse=reuse,
		             scope=scope)
		# Update feature shapes (try at least!)



	def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Network arg_scope.
		"""
		return ssd_arg_scope(weight_decay, data_format=data_format)

	def arg_scope_caffe(self, caffe_scope):
		"""Caffe arg_scope used for weights importing.
		"""
		return ssd_arg_scope_caffe(caffe_scope)

	# ======================================================================= #
	'''
	def update_feature_shapes(self, predictions):
		"""Update feature shapes from predictions collection (Tensor or Numpy
		array).
		"""
		shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
		self.params = self.params._replace(feat_shapes=shapes)
	'''

	def anchors(self, img_shape, dtype=np.float32):
		"""Compute the default anchor boxes, given an image shape.
		"""
		return textbox_achor_all_layers(img_shape,
		                                self.params.feat_shapes,
		                                self.params.anchor_ratios,
		                                self.params.anchor_sizes,
		                                self.params.anchor_steps,
		                                self.params.scales,
		                                0.5,
		                                dtype)

	def bboxes_encode(self,glabels, bboxes, anchors, gxs, gys,
	                  scope='text_bboxes_encode'):
		"""Encode labels and bounding boxes.
		"""
		return textbox_common.tf_text_bboxes_encode(
			glabels, bboxes, anchors, gxs, gys,
			matching_threshold=0.5,
			prior_scaling=self.params.prior_scaling,
			scope=scope)

	def losses(self, logits, localisations,
	           glabels,
	           glocalisations, gscores,
	           match_threshold=0.5,
	           negative_ratio=3.,
	           alpha=0.2,
	           label_smoothing=0.,
	           batch_size=16,
	           scope='txt_losses'):
		"""Define the SSD network losses.
		"""
		return ssd_losses(logits, localisations,
		                  glabels,
		                  glocalisations, gscores,
		                  match_threshold=match_threshold,
		                  negative_ratio=negative_ratio,
		                  alpha=alpha,
		                  label_smoothing=label_smoothing,
		                  batch_size = batch_size,
		                  scope=scope)


def text_multibox_layer(layer,
                        inputs,
                        anchor_size,
                        anchor_ratio,
                        normalization=-1):
	"""
	Construct a multibox layer, return a class and localization predictions.
	The  most different between textbox and ssd is the prediction shape
	where textbox has prediction score shape (48,48,2,6)
	and location has shape (48,48,2,6,12)
	besise,the kernel for every layer same
	"""
	net = inputs
	if normalization > 0:
		net = custom_layers.l2_normalization(net, scaling=True)
	# Number of anchors.
	num_anchors = len(anchor_ratio) + len(anchor_size)
	num_classes = 2
	# Location.

	# location 4+8
	num_prior_per_location = 2 * num_anchors
	num_loc_pred = num_prior_per_location* 12
	# num_loc_pred = num_prior_per_location * 4
    #240/12 = 20 = num_prior_per_location
	# if(layer == 'conv11'):
	#     loc_pred = slim.conv2d(net, num_loc_pred, [1, 1], activation_fn=None, padding = 'VALID',
	#                        scope='conv_loc')
	# else:
	loc_pred = slim.conv2d(net, num_loc_pred, [3, 5], activation_fn=None, padding='SAME', scope='conv_loc')
	# loc_pred = custom_layers.channel_to_last(loc_pred)
	loc_pred = slim.flatten(loc_pred)
	l_shape = loc_pred.shape
	batch_size = l_shape[0]
	loc_pred = tf.reshape(loc_pred, [batch_size, -1, 12])
	# loc_pred = tf.reshape(loc_pred, [1, -1, 2])
	# loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [2, num_anchors, 12])
	# loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [2, num_anchors, 4])
	# Class prediction.
	scores_pred = num_prior_per_location * num_classes
	# scores_pred = num_classes
    # scores_pred = 40
	# if(layer == 'conv11'):
	#     sco_pred = slim.conv2d(net, scores_pred, [1, 1], activation_fn=None, padding = 'VALID',
	#                        scope='conv_cls')
	# else:
	sco_pred = slim.conv2d(net, scores_pred, [3, 5], activation_fn=None, padding='SAME',scope='conv_cls')
	# cls_pred = custom_layers.channel_to_last(cls_pred)
	# sco_pred = tf.reshape(sco_pred, sco_pred.get_shape().as_list()[:-1] + [2, num_anchors, num_classes])
	sco_pred = slim.flatten(sco_pred)
	sco_pred = tf.reshape(sco_pred, [batch_size, -1 ,2])
	return sco_pred, loc_pred


def text_net(inputs,
             feat_layers=TextboxNet.default_params.feat_layers,
             anchor_sizes=TextboxNet.default_params.anchor_sizes,
             anchor_ratios = TextboxNet.default_params.anchor_ratios,
             normalizations=TextboxNet.default_params.normalizations,
             is_training=True,
             dropout_keep_prob=0.5,
             reuse=None,
             scope='text_box_384'):
	end_points = {}
	with tf.compat.v1.variable_scope(scope, 'text_box_300', [inputs], reuse=reuse):  # 300*300 384*383
		# Original VGG-16 blocks.
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')  # 300 384
		end_points['conv1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 150
		# Block 2.
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')  # 150 192
		end_points['conv2'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 75
		# Block 3.
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')  # 75 81
		end_points['conv3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 38
		# Block 4.
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')  # 38 40
		end_point = 'conv4'

		end_points[end_point] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 19
		# Block 5.
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')  # 19
		end_points['conv5'] = net
		net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')  # 19

		# Additional SSD blocks.
		# Block 6: let's dilate the hell out of it!
		net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')  # 19
		end_points['conv6'] = net
		# Block 7: 1x1 conv. Because the fuck.
		net = slim.conv2d(net, 1024, [1, 1], scope='conv7')  # 19
		end_point = 'conv7'

		end_points[end_point] = net

		# Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2
		end_point = 'conv8'
		with tf.compat.v1.variable_scope(end_point):
			net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')

		end_points[end_point] = net  # 10
		end_point = 'conv9'
		with tf.compat.v1.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')

		end_points[end_point] = net # 5
		end_point = 'conv10'
		with tf.compat.v1.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1', padding= 'VALID')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')

		end_points[end_point] = net  # 3
		end_point = 'conv11'
		with tf.compat.v1.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')

		end_points[end_point] = net  #



		end_point = feat_layers[0]
		with tf.compat.v1.variable_scope(end_point):
			net_dilation1 = slim.conv2d(end_points[end_point], 128, [3, 3], stride=1, scope='dilation1')

			# net_dilation2 = custom_layers.pad2d(net, pad=(0, 4))

			net_dilation2 = slim.conv2d(end_points[end_point], 128, [1, 9], padding='SAME', stride=1, scope='dilation2')

			net_dilation3 = slim.conv2d(end_points[end_point], 128, [9, 1], stride=1, padding='SAME', scope='dilation3')
			# net_dilation3 = custom_layers.pad2d(net_dilation3, pad=(4, 0))
			net_inception = tf.concat(values=[net_dilation1, net_dilation2, net_dilation3], axis=3)

		end_points[end_point] = net_inception

		end_point= feat_layers[1]
		with tf.compat.v1.variable_scope(end_point):
			net_dilation1 = slim.conv2d(end_points[end_point], 1024, [1, 1], stride=1, scope='dilation1')

			net_dilation2 = slim.conv2d(end_points[end_point], 1024, [1, 7], stride=1, scope='dilation2')
			# net_dilation2 = custom_layers.pad2d(net_dilation2, pad=(0, 3))

			net_dilation3 = slim.conv2d(end_points[end_point], 1024, [7, 1], stride=1, scope='dilation3')
			# net_dilation3 = custom_layers.pad2d(net_dilation3, pad=(3, 0))

			net_inception = tf.concat([net_dilation1, net_dilation2, net_dilation3], axis=3)

		end_points[end_point] = net_inception


		end_point = 'conv8'
		with tf.compat.v1.variable_scope(end_point):

			net_dilation1 = slim.conv2d(end_points[end_point], 128, [1, 1], stride=1,scope='dilation1')

			net_dilation2 = slim.conv2d(end_points[end_point], 128, [1, 7], stride=1, scope='dilation2')
			# net_dilation2 = custom_layers.pad2d(net_dilation2, pad=(0, 3))

			net_dilation3 = slim.conv2d(end_points[end_point], 128, [7, 1], stride=1, scope='dilation3')
			# net_dilation3 = custom_layers.pad2d(net_dilation3, pad=(3, 0))

			net_inception = tf.concat([net_dilation1, net_dilation2, net_dilation3], axis=3)

		end_points[end_point] = net_inception


		end_point = feat_layers[3]
		with tf.compat.v1.variable_scope(end_point):
			net_dilation1 = slim.conv2d(end_points[end_point], 128, [1, 1], stride=1,scope='dilation1')

			net_dilation2 = slim.conv2d(end_points[end_point], 128, [1, 7], stride=1, scope='dilation2')
			# net_dilation2 = custom_layers.pad2d(net_dilation2, pad=(0, 3))

			net_dilation3 = slim.conv2d(end_points[end_point], 128, [7, 1], stride=1, scope='dilation3')
			# net_dilation3 = custom_layers.pad2d(net_dilation3, pad=(3, 0))
			net_inception = tf.concat([net_dilation1, net_dilation2, net_dilation3], axis=3)

		end_points[end_point] = net_inception # 5

		end_point = 'conv10'
		with tf.compat.v1.variable_scope(end_point):

			net_dilation1 = slim.conv2d(end_points[end_point], 128, [1, 1], stride=1, scope='dilation1')

			net_dilation2 = slim.conv2d(end_points[end_point], 128, [1, 7], stride=1, scope='dilation2')
			# net_dilation2 = custom_layers.pad2d(net_dilation2, pad=(0, 3))

			net_dilation3 = slim.conv2d(end_points[end_point], 128, [7, 1], stride=1, scope='dilation3')
			# net_dilation3 = custom_layers.pad2d(net_dilation3, pad=(3, 0))
			net_inception = tf.concat([net_dilation1, net_dilation2, net_dilation3], axis=3)

		end_points[end_point] = net_inception  # 3


		end_point = 'conv11'
		with tf.compat.v1.variable_scope(end_point):

			net_dilation1 = slim.conv2d(end_points[end_point], 128, [1, 1], stride=1,scope='dilation1')

			net_dilation2 = slim.conv2d(end_points[end_point], 128, [1, 5], stride=1, scope='dilation2')
			# net_dilation2 = custom_layers.pad2d(net_dilation2, pad=(0, 2))

			net_dilation3 = slim.conv2d(end_points[end_point], 128, [5, 1], stride=1, scope='dilation3')
			# net_dilation3 = custom_layers.pad2d(net_dilation3, pad=(2, 0))
			net_inception = tf.concat([net_dilation1, net_dilation2, net_dilation3], axis=3)

		end_points[end_point] = net_inception  # 1

		# Prediction and localisations layers.
		predictions = []
		logits = []
		localisations = []
		for i, layer in enumerate(feat_layers):
			with tf.compat.v1.variable_scope(layer + '_box'):
				p, loc = text_multibox_layer(layer,
				                             end_points[layer],
				                             anchor_sizes[i],
				                             anchor_ratios[i],
				                             normalizations[i])
			prediction_fn = slim.softmax
			predictions.append(prediction_fn(p))
			logits.append(p)
			localisations.append(loc)

		return predictions, localisations, logits, end_points


## produce anchor for one layer
# each feature point has 12 default textboxes(6 boxes + 6 offsets boxes)
# aspect ratios = (1,2,3,5,1/2,1/3, 1/5)
# feat_size :
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# pool6 ==> 1 x 1
"""Computer TextBoxes++ default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
"""


def textbox_anchor_one_layer(img_shape,
                             feat_size,
                             ratios,
                             size,
                             step,
                             scale,
                             offset=0.5,
                             dtype=np.float32):
	# Follow the papers scheme
	# 12 ahchor boxes with out sk' = sqrt(sk * sk+1)
	# size_h = img_shape[0] / 384
	# size_w = img_shape[1] / 384
	#
	# feat_size_h = feat_size[0] * size_h
	# feat_size_w = feat_size[1] * size_w

	y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]] + 0.5

	y_offset = (y.astype(dtype) + 0.5) * step / img_shape[0]
	y = y.astype(dtype) * step / img_shape[0]
	x = x.astype(dtype) * step / img_shape[1]
	x_offset = x

	#38,38,2 origin and offset
	x_out = np.stack((x, x_offset), -1)
	y_out = np.stack((y, y_offset), -1)
	# add dims
	y_out = np.expand_dims(y_out, axis=-1)
	x_out = np.expand_dims(x_out, axis=-1)

	#
	num_anchors = len(ratios) + len(size)
	h = np.zeros((num_anchors,), dtype=dtype)
	w = np.zeros((num_anchors,), dtype=dtype)
	# first prior
	h[0] = size[0] / img_shape[0]
	w[0] = size[0] / img_shape[1]
	di = 1
	if len(size) > 1:
		h[1] = math.sqrt(size[0] * size[1]) / img_shape[0]
		w[1] = math.sqrt(size[0] * size[1]) / img_shape[1]
		di += 1

	for i, r in enumerate(ratios):
		h[i+di] = size[0] / img_shape[0] /math.sqrt(r)
		w[i+di] = size[0] / img_shape[1] * math.sqrt(r)
		# h[i] = scale / math.sqrt(r) / feat_size[0]
		# w[i] = scale * math.sqrt(r) / feat_size[1]
	xmin = x_out - w/2
	ymin = y_out - h/2
	xmax = x_out + w/2
	ymax = y_out + h/2

	xmin = xmin.reshape([xmin.shape[0], xmin.shape[1], -1], order='F').reshape(-1)
	ymin = ymin.reshape([ymin.shape[0], ymin.shape[1], -1], order='F').reshape(-1)
	xmax = xmax.reshape([xmax.shape[0], xmax.shape[1], -1], order='F').reshape(-1)
	ymax = ymax.reshape([ymax.shape[0], ymax.shape[1], -1], order='F').reshape(-1)

	return xmin, ymin, xmax, ymax


## produce anchor for all layers
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
	"""Defines the VGG arg scope.

	Args:
	  weight_decay: The l2 regularization coefficient.

	Returns:
	  An arg_scope.
	"""
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
	                    activation_fn=tf.nn.relu,
	                    weights_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay)),
	                    weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
	                    biases_initializer=tf.compat.v1.zeros_initializer()):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
		                    padding='SAME',
		                    data_format=data_format):
			with slim.arg_scope([custom_layers.pad2d,
			                     custom_layers.l2_normalization,
			                     custom_layers.channel_to_last],
			                    data_format=data_format) as sc:
				return sc


def textbox_achor_all_layers(img_shape,
                             layers_shape,
                             anchor_ratios,
                             anchor_sizes,
                             anchor_steps,
                             scales,
                             offset=0.5,
                             dtype=np.float32):
	"""
	Compute anchor boxes for all feature layers.
	"""
	layers_anchors = []
	for i, s in enumerate(layers_shape):
		anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
		                                         anchor_ratios[i],
		                                         anchor_sizes[i],
		                                         anchor_steps[i],
		                                         scales[i],
		                                         offset=offset, dtype=dtype)
		layers_anchors.append(anchor_bboxes)
	return layers_anchors


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
	"""Caffe scope definition.

	Args:
	  caffe_scope: Caffe scope object with loaded weights.

	Returns:
	  An arg_scope.
	"""
	# Default network arg scope.
	with slim.arg_scope([slim.conv2d],
	                    activation_fn=tf.nn.relu,
	                    weights_initializer=caffe_scope.conv_weights_init(),
	                    biases_initializer=caffe_scope.conv_biases_init()):
		with slim.arg_scope([slim.fully_connected],
		                    activation_fn=tf.nn.relu):
			with slim.arg_scope([custom_layers.l2_normalization],
			                    scale_initializer=caffe_scope.l2_norm_scale_init()):
				with slim.arg_scope([slim.conv2d, slim.max_pool2d],
				                    padding='SAME') as sc:
					return sc


# =========================================================================== #
# Text loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,glabels,
               glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=0.2,
               label_smoothing=0.,
               batch_size=16,
               scope=None):
	'''Loss functions for training the text box network.
	Arguments:
	  logits: (list of) predictions logits Tensors;                x
	  localisations: (list of) localisations Tensors;              l
	  glocalisations: (list of) groundtruth localisations Tensors; g
	  gscores: (list of) groundtruth score Tensors;                c
	'''
	# from ssd loss
	with tf.compat.v1.name_scope(scope, 'txt_losses'):
		lshape = tfe.get_shape(logits[0], 5)
		num_classes = lshape[-1]
		batch_size = batch_size

		l_cross_pos = []
		l_cross_neg = []
		l_loc = []

		# Flatten out all vectors!
		flogits = logits
		fgscores = gscores
		flocalisations = localisations
		fglocalisations = glocalisations
		fglabels = glabels
		# for i in range(len(logits)):
		# 	flogits.append(tf.reshape(logits[i], [-1, num_classes]))
		# 	fgscores.append(tf.reshape(gscores[i], [-1]))
		# 	fglabels.append(tf.reshape(glabels[i], [-1]))
		# 	flocalisations.append(tf.reshape(localisations[i], [-1, 12]))
		# 	fglocalisations.append(tf.reshape(glocalisations[i], [-1, 12]))
		# And concat the crap!
		glabels = tf.concat(fglabels, axis=1)
		logits = tf.concat(flogits, axis=1)  # x
		gscores = tf.concat(fgscores, axis=1)  # c
		localisations = tf.concat(flocalisations, axis=1)  # l
		glocalisations = tf.concat(fglocalisations, axis=1)  # g
		dtype = logits.dtype

		# Compute positive matching mask...
		pmask = gscores > match_threshold  # positive mask
		# pmask = tf.concat(axis=0, values=[pmask[:tf.argmax(gscores, axis=0)], [True], pmask[tf.argmax(gscores, axis=0) + 1:]])

		ipmask = tf.cast(pmask, tf.int32)  # int positive mask
		fpmask = tf.cast(pmask, dtype)  # float positive mask
		n_positives = tf.reduce_sum(input_tensor=fpmask)  # calculate all number

		# Hard negative mining...
		# conf loss ??
		no_classes = tf.cast(pmask, tf.int32)
		predictions = slim.softmax(logits)  #
		nmask = tf.logical_and(tf.logical_not(pmask),
		                       gscores > -0.5)  #
		fnmask = tf.cast(nmask, dtype)
		nvalues = tf.compat.v1.where(nmask,
		                   predictions[:, :, 0],
		                   1. - fnmask)
		nvalues_flat = tf.reshape(nvalues, [-1])
		# Number of negative entries to select.
		max_neg_entries = tf.cast(tf.reduce_sum(input_tensor=fnmask), tf.int32)
		n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
		n_neg = tf.minimum(n_neg, max_neg_entries)

		val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
		max_hard_pred = -val[-1]
		# Final negative mask.
		nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
		fnmask = tf.cast(nmask, dtype)
		inmask = tf.cast(nmask, tf.int32)
		# Add cross-entropy loss.
		# logits [batch_size, num_classes] labels [batch_size] ~ 0,num_class
		with tf.compat.v1.name_scope('cross_entropy_pos'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=glabels)
			loss = tf.compat.v1.div(tf.reduce_sum(input_tensor=loss * fpmask), batch_size, name='value')
			tf.compat.v1.losses.add_loss(loss)
			l_cross_pos.append(loss)

		with tf.compat.v1.name_scope('cross_entropy_neg'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
			                                                      labels=no_classes)
			loss = tf.compat.v1.div(tf.reduce_sum(input_tensor=loss * fnmask), batch_size, name='value')
			tf.compat.v1.losses.add_loss(loss)
			l_cross_neg.append(loss)

		# Add localization loss: smooth L1, L2, ...
		with tf.compat.v1.name_scope('localization'):
			# Weights Tensor: positive mask + random negative.
			weights = tf.expand_dims(alpha * fpmask, axis=-1)
			# localisations = tf.Print(localisations, [localisations, tf.shape(localisations)], "pre is:         ", summarize=20)
			# glocalisations = tf.Print(glocalisations, [glocalisations,  tf.shape(glocalisations)], "gt is :         ",summarize=20)
			loss = custom_layers.abs_smooth(localisations - glocalisations)
			loss = tf.compat.v1.div(tf.reduce_sum(input_tensor=loss * weights), batch_size, name='value')
			tf.compat.v1.losses.add_loss(loss)
			l_loc.append(loss)

		with tf.compat.v1.name_scope('total'):
			total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
			total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
			total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
			total_loc = tf.add_n(l_loc, 'localization')

			# Add to EXTRA LOSSES TF.collection
			tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross_pos)
			tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross_neg)
			tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross)
			tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_loc)

	# with tf.name_scope(scope, 'txt_losses'):
	#     l_cross_pos = []
	#     l_cross_neg = []
	#     l_loc = []
	#     tf.logging.info('logits len:',len(logits), logits)
	#     for i in range(len(logits)):
	#         dtype = logits[i].dtype
	#         with tf.name_scope('block_%i' % i):
	#
	#             # Determine weights Tensor.
	#             pmask = gscores[i] > match_threshold
	#             ipmask = tf.cast(pmask, tf.int32)
	#             fpmask = tf.cast(pmask, dtype)
	#             n_positives = tf.reduce_sum(fpmask)
	#
	#             # Negative mask
	#             # Number of negative entries to select.
	#             n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
	#
	#             nvalues = tf.where(tf.cast(1-ipmask,tf.bool), gscores[i], np.zeros(gscores[i].shape))
	#             nvalues_flat = tf.reshape(nvalues, [-1])
	#             val, idxes = tf.nn.top_k(nvalues_flat, k=n_neg)
	#             minval = val[-1]
	#             # Final negative mask.
	#             nmask = nvalues > minval
	#             fnmask = tf.cast(nmask, dtype)
	#             inmask = tf.cast(nmask, tf.int32)
	#             # Add cross-entropy loss.
	#             with tf.name_scope('cross_entropy_pos'):
	#                 loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
	#                                                                       labels=ipmask)
	#                 loss = tf.losses.compute_weighted_loss(loss, fpmask)
	#                 l_cross_pos.append(loss)
	#
	#             with tf.name_scope('cross_entropy_neg'):
	#                 loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
	#                                                                       labels=inmask)
	#                 loss = tf.losses.compute_weighted_loss(loss, fnmask)
	#                 l_cross_neg.append(loss)
	#
	#             # Add localization loss: smooth L1, L2, ...
	#             with tf.name_scope('localization'):
	#                 # Weights Tensor: positive mask + random negative.
	#                 weights = tf.expand_dims(alpha * fpmask, axis=-1)
	#                 loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
	#                 loss = tf.losses.compute_weighted_loss(loss, weights)
	#                 l_loc.append(loss)
	#
	#     # Additional total losses...
	#     with tf.name_scope('total'):
	#         total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
	#         total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
	#         total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
	#         total_loc = tf.add_n(l_loc, 'localization')
	#
	#         # Add to EXTRA LOSSES TF.collection
	#         tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
	#         tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
	#         tf.add_to_collection('EXTRA_LOSSES', total_cross)
	#         tf.add_to_collection('EXTRA_LOSSES', total_loc)

	# with tf.name_scope(scope, 'txt_losses'):
	#         l_cross_pos = []
	#         l_cross_neg = []
	#         l_loc = []
	#         for i in range(len(logits)):
	#             dtype = logits[i].dtype
	#             with tf.name_scope('block_%i' % i):
	#                 # Sizing weight...
	#
	#                 wsize = tfe.get_shape(logits[i], rank=5)
	#                 wsize = wsize[1] * wsize[2] * wsize[3]
	#
	#                 # Positive mask.
	#                 pmask = gscores[i] > match_threshold
	#                 fpmask = tf.cast(pmask, dtype)
	#                 ipmask = tf.cast(fpmask, 'int32')
	#                 n_positives = tf.reduce_sum(fpmask)
	#
	#                 # Negative mask.
	#                 no_classes = tf.cast(pmask, tf.int32)
	#                 predictions = slim.softmax(logits[i])
	#                 nmask = tf.logical_and(tf.logical_not(pmask),
	#                                        gscores[i] > -0.5)
	#                 fnmask = tf.cast(nmask, dtype)
	#                 print('nmask',nmask)
	#                 print('preditions', predictions)
	#                 predictions_low = predictions[:, :, :, :, :, 0]
	#                 print('fnmask', 1.-fnmask)
	#                 nvalues = tf.where(nmask,
	#                                    predictions_low,
	#                                    1. - fnmask)
	#                 nvalues_flat = tf.reshape(nvalues, [-1])
	#                 # Number of negative entries to select.
	#                 n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
	#                 n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
	#                 n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
	#                 max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
	#                 n_neg = tf.minimum(n_neg, max_neg_entries)
	#
	#                 val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
	#                 max_hard_pred = -val[-1]
	#                 # Final negative mask.
	#                 nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
	#                 fnmask = tf.cast(nmask, dtype)
	#
	#                 # Add cross-entropy loss.
	#                 with tf.name_scope('cross_entropy_pos'):
	#                     fpmask = wsize * fpmask
	#                     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
	#                                                                           labels=ipmask)
	#                     loss = tf.losses.compute_weighted_loss(loss, fpmask)
	#                     l_cross_pos.append(loss)
	#
	#                 with tf.name_scope('cross_entropy_neg'):
	#                     fnmask = wsize * fnmask
	#                     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
	#                                                                           labels=no_classes)
	#                     loss = tf.losses.compute_weighted_loss(loss, fnmask)
	#                     l_cross_neg.append(loss)
	#
	#                 # Add localization loss: smooth L1, L2, ...
	#                 with tf.name_scope('localization'):
	#                     # Weights Tensor: positive mask + random negative.
	#                     weights = tf.expand_dims(alpha * fpmask, axis=-1)
	#                     loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
	#                     loss = tf.losses.compute_weighted_loss(loss, weights)
	#                     l_loc.append(loss)
	#
	#         # Additional total losses...
	#         with tf.name_scope('total'):
	#             total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
	#             total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
	#             total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
	#             total_loc = tf.add_n(l_loc, 'localization')
	#
	# # Add to EXTRA LOSSES TF.collection
	# tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
	# tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
	# tf.add_to_collection('EXTRA_LOSSES', total_cross)
	# tf.add_to_collection('EXTRA_LOSSES', total_loc)
