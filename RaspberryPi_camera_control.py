# This is the code for camera control on Raspberry Pi

# Camera description
# RGB camera module : Raspberry Pi High Quality Camera
# IR camera moduel : FLIR Lepton 3.5 camera module

# Core idea about this camera control module:
# RGB camera:
# Since RGB camera will be connect with the Raspberry Pi directly
# Reference website: 
# https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/
# IR camera:
# The RGB camera will be connected with the Raspberry Pi through the USB
# Even thought it is the FlIR Lepton Camera module, as long as it connect to Raspberry Pi through the USE
# It has to use OpenCV to access the camera
# Reference website:
# 1. Raspberry Pi tutorial on using a USB camera to display and record videos with Python
# https://medium.com/@ammani/python-usb-camera-tutorial-for-raspberry-pi-4-3098678a2464  
# 2. How to install OpenCV RaspberryPi
# Install OpenCV 4 on Raspberry Pi 4 and Raspbian Buster
# https://pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/


# Work Flow
# 1. Access the FLIR Lepton (Using Python)
# 2. Access the Raspberry Pi High Quality Camera (Using Python)
# 3. Be able to control the two cameras at the same time to take obtain the images
# 4. Be able to save the matched images
# 5. Be able to obtain the object temperature (radiometric information)

#%%
# import packages
import cv2 # load OpenCV




