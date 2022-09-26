# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:26:39 2022.9.6

@author: Chenlizhou
"""

import numpy as np
import matplotlib.pyplot as plt 
import time
import os

from src.autoencoder_project.config import Config
from src.autoencoder_project.train import Autoencoder_training

def process_fruit_images_using_autoencoder():
	img_path = './static/images/test.png'
	# Process the image using the network. Print the time used.
	start=time.process_time()
	config = Config(config_path='../autoencoder_project/config/config.yml')
	autoencoder_training = Autoencoder_training(config)
	autoencoder_training.test(input_path=img_path, save_result_path='./static/images/test2.png')
	time_used = (time.process_time() - start)
	print("Time used:",time_used)

	return 0

if __name__ == '__main__':
    process_fruit_images_using_autoencoder()