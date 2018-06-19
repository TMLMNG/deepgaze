#!/usr/bin/env python

from Tkinter import Tk
import tkFileDialog as fd
import re

import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() # Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) # Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))

# Simple file dialog to choose a picture
Tk().withdraw()
file_name = fd.askopenfilename()
print("\nPath:\n" + file_name + '\n')
image_name = os.path.basename(file_name)

# Extract filename data (TODO)
verti_angle = re.search(r'(?<=(-|\+))\w+', image_name).group(0)
horiz_angle = re.search(r'(?<=(-|\+))\w+', image_name).group(0)
print("Image Vertical Angle:   " + verti_angle)
print("Image Horizontal Angle: " + horiz_angle)

# Process image with OpenCV2
print("\nProcessing image ..... " + image_name + '\n')
image = cv2.imread(file_name) # Read the image with OpenCV

# Get the angles for roll, pitch and yaw
roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN

print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + ", " + str(pitch[0,0,0]) + ", " + str(yaw[0,0,0])  + "]\n")