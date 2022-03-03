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

# Note :
# Is it necessary to install OpenCV on Raspberry Pi ?
# It is dedicated for installing OpenCV in Ubuntu and Debian OS. 
# opencv-python is the OpenCV library available as a wrapper with bindings for python. The link also shows how to install OpenCV in Ubuntu OS


# Work Flow
# 1. Access the FLIR Lepton 3.5 camera USB (Using OpenCV)
# 2. Access the Raspberry Pi High Quality Camera (Using OpenCV)
# 3. Be able to control the two cameras at the same time to take obtain the images
# 4. Be able to save the matched images
# 5. Be able to obtain the object temperature (radiometric information)

#%%
# import packages
import cv2 # load OpenCV
import  matplotlib.pyplot as plt # optional package to import, in order to visualize an image
import os
# Now, I don't have the FLIR Lepton yet, in order to test if the work pipeline works or not.
# Here I choose a USB Digital Webcam to test, if I am able to use the OpenCV to control the USB camera

# 1. Access the USB camera (Using OpenCV)
#%%
# define a function to take images
def take_images_USB(camera_index, output_path, image_name):
    """
    Function to use take images through USB camera 
    Arguments : 
    Input:
    camera_index: int
        the index number of the camera which used to connect to the PC/ Raspberry

    output_path: string
        the working directory for the image that you want to save.

    image_name: string
        the name of the image that you obtained from the USB camera.

    Output:
    Image in the defined working directory.
    """

    # connecting to the USB camera
    camera_USB = cv2.VideoCapture(camera_index) # define the camera as index number
    # get a frame from the capture device
    ret, image =  camera_USB.read()
    # make sure if the camera connected
    # if return value is true then the camera connected, otherwise it is not.
    if ret is False:
        print("The USB camera is OFF")
    """
    Here is the optional to show the video capture images or not.
    It is not necessary to use into data collection, but it is important for the camera test.
    """
    # show the video capture images
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    # create the output directory, if not present then create one
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # save the images
    cv2.imwrite(os.path.join(output_path, image_name), image)
    # Turn off the camera 
    camera_USB.release()
#%% Function test for Image 
USB_camera_index = 0 # Here the USB camera index value is 0, it could be other index values. 
image_save_path = r"/Users/qianchengsun/Desktop/Raspberry_pi_image_collect"
image_file_save_name = "Webcam.jpg"

take_images_USB(camera_index= USB_camera_index,
            output_path= image_save_path,
            image_name= image_file_save_name)

# %% Rendering in Real Time
def real_time_video_USB(camera_index, quit_key):
    """
    Function to obtain the real time video/images through USB camera 
    Arguments : 
    Input:
    camera_index: int
        the index number of the camera which used to connect to the PC/ Raspberry

    quit_key: string
        the hit key to shut down the program
    """
    # Connect to webcam
    camera_USB = cv2.VideoCapture(camera_index)
    # Loop through every frame until we close our webcam
    while camera_USB.isOpened():
        ret, image = camera_USB.read()
        if ret is False:
            print("The USB camera is OFF")
        # show image
        cv2.imshow("webcam", image)
        # effectively checking whether or not we're hitting anything on our keyboard
        # cv2.waitKey(1) gives us a chance to hit a key on our keyboard
        # 0xFF == ord("q") unpacks what is actually being hit

        # if the key that hit is "q", we are going to break out of our while loop
        """
        Check whether "q" has been hit and stops the loop 
        """
        if cv2.waitKey(1) & 0xFF == ord(quit_key):
            break
    camera_USB.release()
    cv2.destroyAllWindows()
# %% Fuction test to rendering the video in real time
real_time_video_USB(camera_index= USB_camera_index,
                    quit_key= "q")
# %%
