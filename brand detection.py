#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:49:13 2021

@author: RishiSingh
"""

cog_key = 'b8945e3a82644d6cbfbe175fd482e9b2'
cog_endpoint = 'https://n0834113t4.cognitiveservices.azure.com/'
cog_region = 'uksouth'

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os


def brand_detection(url):
    import cv2
    import numpy as np
    import requests
    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))
    remote_image_url = url
    # Select the visual feature(s) you want
    remote_image_features = ["brands"]
    # Call API with URL and features
    detect_brands_results_remote = computervision_client.analyze_image(remote_image_url, remote_image_features)  
    print("Detecting brands in remote image: ")
    if len(detect_brands_results_remote.brands) == 0:
        print("No brands detected.")
    else:
        for brand in detect_brands_results_remote.brands:
            print("'{}' brand detected with confidence {:.1f}% at location {}, {}, {}, {}".format( \
        brand.name, brand.confidence * 100, brand.rectangle.x, brand.rectangle.x + brand.rectangle.w, \
        brand.rectangle.y, brand.rectangle.y + brand.rectangle.h))
    resp = requests.get(remote_image_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Blue color in BGR
    color = (255, 0, 0)  
    # Line thickness of 2 px
    thickness = 2
    start_point = (brand.rectangle.x, brand.rectangle.y)
    end_point = ((brand.rectangle.x+brand.rectangle.w),(brand.rectangle.y + brand.rectangle.h))
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    window_name = 'Image'
    thickness = 2
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    


url = "https://cdn.mos.cms.futurecdn.net/uWjEogFLUTBc8mSvagdiuP-970-80.jpg"
brand_detection(url)
