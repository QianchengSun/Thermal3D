#%%
import copy
from re import T
from sys import flags
import numpy as np
import os
import json
import cv2
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
from numpy.lib import save
from plyfile import PlyData
import os
import pandas as pd
from tqdm import tqdm
#%% RGB_camera_calibration
def RGB_camera_calibration(RGB_image_folder,
                            RGB_prefix,
                            RGB_image_format,
                            RGB_output_dir,
                            save_value_switch = True):
    """
    Camera calibration for RGB camera. 
    Output: 
        RGB_camera_matrix : numpy array, camera intrinsic matrix, result for camera calibration (shape = (3,3))
                            will be saved as .npy file by setting RGB_matrix_switch 

        RGB_camera_dist : numpy array, camera distortion coeffecients, result for camera calibration (shape = (1,5))
                        will be saved as .npy file by setting RGB_matrix_switch

        RGB_detected_points_switch : list, will be saved as .npz file by setting RGB_detected_points_switch

    Arguments :
        RGB_image_folder : str , the working directory for RGB_image which is prepared for RGB_camera_calibration

        RGB_prefix : str , the name of the RGB_calibration_images

        RGB_format : str, the format of the RGB_calibration_images, such as "JPG", "JPEG"
    
    Switch:
        RGB_matrix_switch : bool, 
                        An ON / OFF switch for saving the result of IR camera calibration,
                        the default value is ON

        RGB_detected_points_switch : bool,
                        An ON / OFF switch for saving detected points of IR camera calibration,
                        the default value is ON
    Other Arguments:
            RGB_output_dir : str, the working directory which save all the results
    
    """
    print("RGB camera calibration is started ....")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 2000    # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 100000   # maxArea may be adjusted to suit for your experiment

    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    # Original blob coordinates, supposing all blobs are of z-coordinates 0
    # And, the distance between every two neighbour blob circle centers is 72 centimetres
    # In fact, any number can be used to replace 72.
    # Namely, the real size of the circle is pointless while calculating camera calibration parameters.
    objp = np.zeros((44, 3), np.float32)
    objp[0]  = (0  , 0  , 0)
    objp[1]  = (0  , 10.16 , 0)
    objp[2]  = (0  , 20.32, 0)
    objp[3]  = (0  , 30.48, 0)
    objp[4]  = (5.08 , 5.08 , 0)
    objp[5]  = (5.08 , 15.24, 0)
    objp[6]  = (5.08 , 25.4, 0)
    objp[7]  = (5.08 , 35.56, 0)
    objp[8]  = (10.16 , 0  , 0)
    objp[9]  = (10.16 , 10.16 , 0)
    objp[10] = (10.16 , 20.32, 0)
    objp[11] = (10.16 , 30.48, 0)
    objp[12] = (15.24, 5.08,  0)
    objp[13] = (15.24, 15.24, 0)
    objp[14] = (15.24, 25.4, 0)
    objp[15] = (15.24, 35.56, 0)
    objp[16] = (20.32, 0  , 0)
    objp[17] = (20.32, 10.16 , 0)
    objp[18] = (20.32, 20.32, 0)
    objp[19] = (20.32, 30.48, 0)
    objp[20] = (25.4, 5.08 , 0)
    objp[21] = (25.4, 15.24, 0)
    objp[22] = (25.4, 25.4, 0)
    objp[23] = (25.4, 35.56 , 0)
    objp[24] = (30.48, 0  , 0)
    objp[25] = (30.48, 10.16 , 0)
    objp[26] = (30.48, 20.32, 0)
    objp[27] = (30.48, 30.48, 0)
    objp[28] = (35.56, 5.08 , 0)
    objp[29] = (35.56, 15.24, 0)
    objp[30] = (35.56, 25.4, 0)
    objp[31] = (35.56, 35.56, 0)
    objp[32] = (40.64, 0  , 0)
    objp[33] = (40.64, 10.16 , 0)
    objp[34] = (40.64, 20.32, 0)
    objp[35] = (40.64, 30.48, 0)
    objp[36] = (45.72, 5.08 , 0)
    objp[37] = (45.72, 15.24, 0)
    objp[38] = (45.72, 25.4, 0)
    objp[39] = (45.72, 35.56, 0)
    objp[40] = (50.8, 0  , 0)
    objp[41] = (50.8, 10.16 , 0)
    objp[42] = (50.8, 20.32, 0)
    objp[43] = (50.8, 30.48, 0)

    ###################################################################################################

    # Arrays to store object points and image points from all the images.
    ideal_camera_objpoints = [] # 3d point in real world space
    ideal_camera_imgpoints = [] # 2d points in image plane.
     # glob.glob() : obtain the path 
    RGB_images = glob.glob(RGB_image_folder + '/' + RGB_prefix + '*.' + RGB_image_format)
    RGB_images = natsorted(RGB_images)

    count = 0
    for fname in tqdm(RGB_images):
        # print("read")
        # cv2.imread() : read image 
        ideal_camera_img = cv2.imread(fname)
        # cv2.cvtColor() : convert RGB image into grayscale image
        ideal_camera_gray = cv2.cvtColor(ideal_camera_img, cv2.COLOR_BGR2GRAY)

        # cv2.bitwise_not() : revise the image pixel from black to white (white to black)
        ideal_camera_gray = cv2.bitwise_not(ideal_camera_gray)
        # detect blobs
        blobs1 = blobDetector.detect(ideal_camera_gray)
        # detect circle pattern

        # cv2.findCirclesGrid() : find centers in the grid of circles
        isFound, ideal_camera_corners = cv2.findCirclesGrid(image = ideal_camera_gray,   # grid view of input circles; it must be an 8-bit grayscale or color image
                                            patternSize = (4,11),  # number oof circles per row and column (patternSize = Size(points_per_row, points_per_column))
                                            flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, 
                                            blobDetector = blobDetector)  # feature detector that finds blobs like dark circles on light background
        # flags: various operation flags can be one of the following values:
        #       CALIB_CB_ASYMMETRIC_GRID : uses asymmetric pattern of circles
        #       CALIB_CB_SYMMETRIC_GRID : uses symmetric pattern of circles
        #       CALIB_CB_CLUSTERING : uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter

        # print(isFound)

        if isFound:
            count += 1
            ideal_camera_objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            
            # cv2.cornerSubPix() : Refines the corner locations
            ideal_camera_corners2 = cv2.cornerSubPix(image = ideal_camera_gray,  # input single-channel, 8 bit or float image
                                        corners = ideal_camera_corners, # Initial coordinates of the input corners and refined coordinates provide for output
                                        winSize = (11,11), # half of the side length of the search window. 
                                        # if winSize = Size(5,5), then a (5*2+1) * (5*2+1) = 11 * 11 search window is used
                                        zeroZone = (-1,-1), # half of the size of dead region in the middle of the search zone over which the summation in the formula below is not done.
                                        # It is used sometimes to avoid possible singularities of the autocorrelation matrix.
                                        # the value oof (-1, -1) indicates that there is no such a size.
                                        criteria = criteria)  # Criteria for termination of the iterative process of corner refinement.
            ideal_camera_imgpoints.append(ideal_camera_corners2)
            
            # Draw and display the circles and centers.

            # cv2.drawChessboardCorners() : Renders the dected chessboard corners
            ideal_camera_img = cv2.drawChessboardCorners(image = ideal_camera_img, # Destination image. It must be an 8-bit color image
                                            patternSize = (4,11), # Number of inner corners per a chessboard row and column
                                            corners = ideal_camera_corners2, # Array of detected corners, the output of findChessboardCorners
                                            patternWasFound = isFound) # Parameters indicating whether thhe complete board was found or not (bool value)

            # cv2.drawKeypoints() : Draw keypoints
            ideal_camera_img = cv2.drawKeypoints(image = ideal_camera_img,  # source image
                                    keypoints = blobs1, # keypoints from the source image
                                    outImage = np.array([]), # output image. Depends on the flags value defining what is drawn in the output image
                                    color = (0,0,255), # color of keypoints
                                    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # For each keypoint the circle around keypoint with keypoint size and orientation will be drawn

            #imS = cv2.resize(img, (np.int(img.shape[1]/5), np.int(img.shape[0]/5)))
            cv2.imshow('img', ideal_camera_img)
            cv2.waitKey(2)

    # When everything done
    cv2.destroyAllWindows()


    # cv2.calibrateCamera() : Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
    ret, ideal_camera_mtx, ideal_camera_dist, ideal_camera_rvecs, ideal_camera_tvecs = cv2.calibrateCamera(objectPoints = ideal_camera_objpoints, # a vector of vectors of calibration pattern points in the calibration pattern coordinate space
                                                    imagePoints = ideal_camera_imgpoints, # a vector of vectors of the projections of calibration pattern points
                                                    imageSize = ideal_camera_gray.shape[::-1], # size of the image used only to initialize the camera intrinsic matrix
                                                    cameraMatrix = None, # input/output 3*3 floating-point camera intrinsic matrix
                                                    distCoeffs = None) # input/output vector of distortion coefficients
    if not os.path.exists(RGB_output_dir):
        os.mkdir(RGB_output_dir)

    if save_value_switch:
        np.save(os.path.join(RGB_output_dir, "RGB_camera_mtx"), ideal_camera_mtx)
        np.save(os.path.join(RGB_output_dir, "RGB_camera_dist"), ideal_camera_dist)
        np.savez(os.path.join(RGB_output_dir, "RGB_detected_points"), ideal_camera_imgpoints)

    print("RGB camera calibration is finished")

    # Here add one print to make sure all the images are recognized 
    if len(RGB_images) > len(ideal_camera_imgpoints) :
        print("NOT ALL input RGB calibration images are read, please remove not recognized RGB calibration images")
    else:
        print("ALL the input RGB calibration images are used to calibrated")

    return ideal_camera_mtx, ideal_camera_dist, ideal_camera_imgpoints

#%% IR camera calibration

def IR_camera_calibration(IR_image_folder, 
                            IR_prefix, 
                            IR_image_format, 
                            SCALE, 
                            IR_output_dir, 
                            save_value_switch = True):
    """
    Camera calibration for thermal camera. 

    Output: 
        IR_camera_matrix : numpy array, camera intrinsic matrix, result for camera calibration (shape = (3,3))
                            will be saved as .npy file by setting IR_matrix_switch 

        IR_camera_dist : numpy array, camera distortion coeffecients, result for camera calibration (shape = (1,5))
                        will be saved as .npy file by setting IR_matrix_switch

        IR_detected_points_switch : list, will be saved as .npz file by setting IR_detected_points_switch

    Arguments :
        IR_image_folder : str , the working directory for IR_image which is prepared for IR_camera_calibration

        IR_prefix : str , the name of the IR_calibration_images

        IR_format : str, the format of the IR_calibration_images, such as "JPG", "JPEG"
    
    Switch:
        IR_matrix_switch : bool, 
                        An ON / OFF switch for saving the result of IR camera calibration,
                        the default value is ON

        IR_detected_points_switch : bool,
                        An ON / OFF switch for saving detected points of IR camera calibration,
                        the default value is ON
    Other Arguments:
        SCALE : int or float number 
                The SCALE value will apply directly on the original IR image in order to obtain a good IR camera calibration result.

        IR_output_dir : str, the working directory which save all the results
    
    """
    print("IR camera calibration is started ....")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 2000     # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 1000000  # maxArea may be adjusted to suit for your experiment

    # # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.85

    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    # Original blob coordinates, supposing all blobs are of z-coordinates 0
    # And, the distance between every two neighbour blob circle centers is 72 centimetres
    # In fact, any number can be used to replace 72.
    # Namely, the real size of the circle is pointless while calculating camera calibration parameters.
    objp = np.zeros((44, 3), np.float32)
    objp[0]  = (0  , 0  , 0)
    objp[1]  = (0  , 10.16 , 0)
    objp[2]  = (0  , 20.32, 0)
    objp[3]  = (0  , 30.48, 0)
    objp[4]  = (5.08 , 5.08 , 0)
    objp[5]  = (5.08 , 15.24, 0)
    objp[6]  = (5.08 , 25.4, 0)
    objp[7]  = (5.08 , 35.56, 0)
    objp[8]  = (10.16 , 0  , 0)
    objp[9]  = (10.16 , 10.16 , 0)
    objp[10] = (10.16 , 20.32, 0)
    objp[11] = (10.16 , 30.48, 0)
    objp[12] = (15.24, 5.08,  0)
    objp[13] = (15.24, 15.24, 0)
    objp[14] = (15.24, 25.4, 0)
    objp[15] = (15.24, 35.56, 0)
    objp[16] = (20.32, 0  , 0)
    objp[17] = (20.32, 10.16 , 0)
    objp[18] = (20.32, 20.32, 0)
    objp[19] = (20.32, 30.48, 0)
    objp[20] = (25.4, 5.08 , 0)
    objp[21] = (25.4, 15.24, 0)
    objp[22] = (25.4, 25.4, 0)
    objp[23] = (25.4, 35.56 , 0)
    objp[24] = (30.48, 0  , 0)
    objp[25] = (30.48, 10.16 , 0)
    objp[26] = (30.48, 20.32, 0)
    objp[27] = (30.48, 30.48, 0)
    objp[28] = (35.56, 5.08 , 0)
    objp[29] = (35.56, 15.24, 0)
    objp[30] = (35.56, 25.4, 0)
    objp[31] = (35.56, 35.56, 0)
    objp[32] = (40.64, 0  , 0)
    objp[33] = (40.64, 10.16 , 0)
    objp[34] = (40.64, 20.32, 0)
    objp[35] = (40.64, 30.48, 0)
    objp[36] = (45.72, 5.08 , 0)
    objp[37] = (45.72, 15.24, 0)
    objp[38] = (45.72, 25.4, 0)
    objp[39] = (45.72, 35.56, 0)
    objp[40] = (50.8, 0  , 0)
    objp[41] = (50.8, 10.16 , 0)
    objp[42] = (50.8, 20.32, 0)
    objp[43] = (50.8, 30.48, 0)
    # Arrays to store object points and image points from all the images.
    thermal_camera_objpoints = [] # 3d point in real world space
    thermal_camera_imgpoints = [] # 2d points in imag
    # glob.glob() : obtain the path 
    IR_images = glob.glob(IR_image_folder + '/' + IR_prefix + '*.' + IR_image_format)
    IR_images = natsorted(IR_images)

    # read image 
    count = 0
    for fname in tqdm(IR_images):
        # print("read")
        # cv2.imread() : read image 
        IR_img = cv2.imread(fname)
        # find the width and height for IR image to do scale 
        width = int(IR_img.shape[1] * SCALE)
        height = int(IR_img.shape[0] * SCALE)
        dim = (width, height)

        # cv2.resize() : resize the image size 
        IR_img = cv2.resize(src = IR_img, 
                        dsize = dim, 
                        interpolation = cv2.INTER_LANCZOS4) # has to use cv2.INTER_LANCZOS4 to do interpolation for IR image
                        # the result for the scaled image will be easy to detected by using blob detector
        # cv2.cvtColor() : convert RGB image into grayscale image
        gray = cv2.cvtColor(IR_img, cv2.COLOR_BGR2GRAY)

        # detect blobs
        blobs2 = blobDetector.detect(gray,None)
        # detect circle pattern
        # cv2.findCirclesGrid() : find centers in the grid of circles
        isFound, thermal_camera_corners = cv2.findCirclesGrid(image = gray,   # grid view of input circles; it must be an 8-bit grayscale or color image
                                            patternSize = (4,11),  # number oof circles per row and column (patternSize = Size(points_per_row, points_per_column))
                                            flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, 
                                            blobDetector = blobDetector)  # feature detector that finds blobs like dark circles on light background
        # flags: various operation flags can be one of the following values:
        #       CALIB_CB_ASYMMETRIC_GRID : uses asymmetric pattern of circles
        #       CALIB_CB_SYMMETRIC_GRID : uses symmetric pattern of circles
        #       CALIB_CB_CLUSTERING : uses a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter

        # print(isFound)

        if isFound:
            count = count + 1
            thermal_camera_objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

            # cv2.cornerSubPix() : Refines the corner locations
            thermal_camera_corners2 = cv2.cornerSubPix(image = gray,  # input single-channel, 8 bit or float image
                                        corners = thermal_camera_corners, # Initial coordinates of the input corners and refined coordinates provide for output
                                        winSize = (11,11), # half of the side length of the search window. 
                                        # if winSize = Size(5,5), then a (5*2+1) * (5*2+1) = 11 * 11 search window is used
                                        zeroZone = (-1,-1), # half of the size of dead region in the middle of the search zone over which the summation in the formula below is not done.
                                        # It is used sometimes to avoid possible singularities of the autocorrelation matrix.
                                        # the value oof (-1, -1) indicates that there is no such a size.
                                        criteria = criteria)  # Criteria for termination of the iterative process of corner refinement.
            thermal_camera_imgpoints.append(thermal_camera_corners2)

            # Draw and display the circles and centers.

            # cv2.drawChessboardCorners() : Renders the dected chessboard corners
            IR_img = cv2.drawChessboardCorners(image = IR_img, # Destination image. It must be an 8-bit color image
                                            patternSize = (4,11), # Number of inner corners per a chessboard row and column
                                            corners = thermal_camera_corners2, # Array of detected corners, the output of findChessboardCorners
                                            patternWasFound = isFound) # Parameters indicating whether thhe complete board was found or not (bool value)

            # cv2.drawKeypoints() : Draw keypoints
            IR_img = cv2.drawKeypoints(image = IR_img,  # source image
                                    keypoints = blobs2, # keypoints from the source image
                                    outImage = np.array([]), # output image. Depends on the flags value defining what is drawn in the output image
                                    color = (0,0,255), # color of keypoints
                                    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # For each keypoint the circle around keypoint with keypoint size and orientation will be drawn

            #imS = cv2.resize(img, (np.int(img.shape[1]/5), np.int(img.shape[0]/5)))
            cv2.imshow('img', IR_img)
            cv2.waitKey(1)

    # When everything done
    cv2.destroyAllWindows()

    # cv2.calibrateCamera() : Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
    ret, thermal_camera_mtx, thermal_camera_dist, thermal_camera_rvecs, thermal_camera_tvecs = cv2.calibrateCamera(objectPoints = thermal_camera_objpoints, # a vector of vectors oof calibration pattern points in the calibration pattern coordinate space
                                                    imagePoints = thermal_camera_imgpoints, # a vector of vectors of the projections of calibration pattern points
                                                    imageSize = gray.shape[::-1], # size of the image used only to initialize the camera intrinsic matrix
                                                    cameraMatrix = None, # input/output 3*3 floating-point camera intrinsic matrix
                                                    distCoeffs = None) # input/output vector of distortion coefficients
    # create the output directory if not present
    if not os.path.exists(IR_output_dir):
            os.mkdir(IR_output_dir)
    # IR_matrix_switch default is ON
    if save_value_switch :
        np.save(os.path.join(IR_output_dir, "IR_camera_mtx"), thermal_camera_mtx)
        np.save(os.path.join(IR_output_dir, "IR_camera_dist"), thermal_camera_dist)
        np.savez(os.path.join(IR_output_dir, "IR_detected_points"), thermal_camera_imgpoints)

    print("IR camera calibration is finished.")        

    # Here add one print to make sure all the images are recognized 
    
    if len(IR_images) > len(thermal_camera_imgpoints) :
        print("NOT ALL input IR calibration images are read, please remove not recognized IR calibration images")
    else:
        print("ALL the input IR calibration images are used to calibrated")

    return thermal_camera_mtx, thermal_camera_dist, thermal_camera_imgpoints

#%% Obtain the projective matrix
def obtain_projective_matrix(RGB_detected_points, 
                            IR_detected_points,
                            RGB_camera_matrix, 
                            IR_camera_matrix,
                            RGB_camera_dist,
                            IR_camera_dist, 
                            output_dir,
                            RGB_detected_points_input_dir = None,
                            IR_detected_points_input_dir = None,
                            RGB_camera_matrix_input_dir = None, 
                            IR_camera_matrix_input_dir = None,
                            RGB_camera_dist_input_dir = None,
                            IR_camera_dist_input_dir = None,
                            read_dir_switch = False,
                            save_value_switch = True):

    """
    Obtain the projective matrix by using calibration results and detected points

    Argument : 
        RGB_camera_detected_points : np.array() 
                                    the detected RGB points from RGB camera calibration which is used for finding the projective matrix

        IR_camera_detected_points : np.array() 
                                    the detected IR points from IR camera calibration which is used for finding the projective matrix

        RGB_camera_matrix : np.array()
                            the RGB camera intrinsic matrix which comes from the RGB camera calibration
        
        IR_camera_matrix : np.array()
                            the IR camera intrinsic matrix which is from the IR camera calibration

        RGB_camera_dist : np.array()
                            the RGB camera distortion coeffection vector which is from the RGB camera calibration

        IR_camera_dist : np.array()
                            the IR camera distortion coeffection vector which is from the IR camera calibration

    Output : projective matrix shape (3x3), there is only one projective matrix will be generate

        output_dir : str the output working directory to save the projective matrix as .npy

    Optional arguments :
        read_dir_switch : bool
                        the ON / OFF switch to determine if load the saved detected points
                        default is OFF
        
        save_projective_switch : bool
                        the ON / OFF switch to determine if save the projective matrix or not
                        default is ON

        RGB_detected_points_input_dir : str 
                                        the input work directory for saved RGB detected points

        IR_detected_points_input_dir : str 
                                        the input work directory for saved IR detected points

        RGB_camera_matrix_iput_dir : str
                            the RGB camera intrinsic matrix which comes from the RGB camera calibration
                            the input work directory for saved RGB camera matrix
        
        IR_camera_matrix_input_dir : str
                            the IR camera intrinsic matrix which is from the IR camera calibration
                            the input work directory for saved IR camera matrix

        RGB_camera_dist_iput_dir : str
                            the RGB camera distortion coeffection vector which is from the RGB camera calibration
                            the input work directory for saved RGB camera coefficients

        IR_camera_dist_input_dir : str
                            the IR camera distortion coeffection vector which is from the IR camera calibration
                            the input work directory for saved IR camera coefficients

    """
    print("Obtain the projective matrix between two camera is started...")

    if read_dir_switch :
        # load npy, npz file
        # only .npz file need to add ["arr_0"] in order to read file.
        RGB_detected_points_read_path = np.load(RGB_detected_points_input_dir)["arr_0"]
        IR_detected_points_read_path = np.load(IR_detected_points_input_dir)["arr_0"]

        RGB_camera_matrix_read_path = np.load(RGB_camera_matrix_input_dir)
        RGB_camera_dist_read_path = np.load(RGB_camera_dist_input_dir)

        IR_camera_matrix_read_path = np.load(IR_camera_matrix_input_dir)
        IR_camera_dist_read_path = np.load(IR_camera_dist_input_dir)

        # make sure the shape is fit for the following steps
        RGB_detected_points_shape = RGB_detected_points_read_path.shape
        RGB_detected_points = RGB_detected_points_read_path.reshape(RGB_detected_points_shape[0]* RGB_detected_points_shape[1], RGB_detected_points_shape[3])

        IR_detected_points_shape = IR_detected_points_read_path.shape
        IR_detected_points = IR_detected_points_read_path.reshape(IR_detected_points_shape[0]* IR_detected_points_shape[1], IR_detected_points_shape[3])

        # undistort points
        """
        Here the method for image undistortion is applied by using OpenCV

        cv.undistortPoints() has been applied to obtain the undistortPoints [x', y']
        Instead of obtain the [x", y"], cv.undistortPoints obtain the value [x', y'] directly from [u,v]

        Methodology :

                x" <------  (u - cx) / fx
                y" <------  (v - cy) / fy
                [x',y'] = undistort(x", y", distCoeffs)
                
                [X, Y, W].T <-----  R * [x', y', 1].T
                x <------  X / W
                y <------  Y / W

                ONLY performed if P is specified : 
                u' <------ x * f'x + c'x
                v' <------ y * f'y + c'y
        
        where undistort is an approximate iterative algorithm that estimates the normalized original point coordinates out of the normalized distorted point coordinates

        Reference website :

            https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html
        
        """
        detected_RGB_undist = cv2.undistortPoints(src = RGB_detected_points,
                                                cameraMatrix = RGB_camera_matrix_read_path,
                                                distCoeffs = RGB_camera_dist_read_path)

        detected_IR_undist = cv2.undistortPoints(src = IR_detected_points,
                                                cameraMatrix = IR_camera_matrix_read_path,
                                                distCoeffs = IR_camera_dist_read_path)
        """
        The method for find projective matrix between two camera has been applied here.

        Here use cv.findHomography() to obtain the perspective transformatin between planes.
        
        Methodology: 

            [x1, y1, 1] = H[x2, y2, 1] 

            H = [[h00, h01, h02],
                 [h10, h11, h12],
                 [h20, h21, h22]]
    
        Reference website :

            https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html
        
        """
        projective_matrix = cv2.findHomography(srcPoints = detected_RGB_undist,
                                                dstPoints = detected_IR_undist,
                                                method = 0)[0]
    else:
        N_image = len(RGB_detected_points)
        N_points = len(RGB_detected_points[0])
        # undistort points
        detected_RGB_undist = cv2.undistortPoints(src = np.array(RGB_detected_points).reshape(N_image * N_points, 2),
                                                cameraMatrix = RGB_camera_matrix,
                                                distCoeffs = RGB_camera_dist)

        detected_IR_undist = cv2.undistortPoints(src = np.array(IR_detected_points).reshape(N_image * N_points, 2),
                                                cameraMatrix = IR_camera_matrix,
                                                distCoeffs = IR_camera_dist)

        projective_matrix = cv2.findHomography(srcPoints = detected_RGB_undist,
                                                dstPoints = detected_IR_undist,
                                                method = 0)[0]
        # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    if save_value_switch :
        np.save(os.path.join(output_dir, "projective_matrix"), projective_matrix)

    print("projective matrix is found.")
    return projective_matrix

# %% Obtain the information from OpenMVG Json file
"""
This is the function that used to obtain the information from OpenMVG
"""
def extract_element_from_json(obj, path):
    def extract(obj, path, ind, arr):
        key = path[ind]
        if ind + 1 < len(path):
            if isinstance(obj, dict):
                if key in obj.keys():
                    extract(obj.get(key), path, ind + 1 , arr)
                else:
                    arr.append(None)
            elif isinstance(obj, list):
                if not obj:
                    arr.append(None)
                else: 
                    for item in obj:
                        extract(item, path, ind, arr)
            else:
                arr.append(None)
        if ind + 1 == len(path):
            if isinstance(obj, list):
                if not obj:
                    arr.append(None)
                else:
                    for item in obj:
                        arr.append(item.get(key, None))
            elif isinstance(obj, dict):
                arr.append(obj.get(key, None))
            else:
                arr.append(None)
        return arr
    if isinstance(obj, dict):
        return extract(obj, path, 0 ,[])
    elif isinstance(obj, list):
        outer_arr = []
        for item in obj:
            outer_arr.append(extract(item, path, 0, []))
        return outer_arr

# %% Obtain the information from OpenMVG sfm.json file
def obtain_information_openMVG(input_dir, output_dir, save_value_switch = True):
    """
    The function is developed to obtain the information from sfm.json which pair to the OpenMVG method

    intrinsic information : camera intrinsic matrix, camera distortion coeffients,

    extrinsic information : rotation matrix, translation vector for each view.
    
    Arguments :
            input_dir : str,
                        the working directory which is used to save the sfm.json file.
            
            output_dir : str,
                        the working directory which will save the information result
            
            save_information_switch : bool
                        the ON / OFF switch to save the information into computer
                        default is ON


    Output:
            rotation_openMVG : numpy array,
                                the rotation matrix for each view which used in OpenMVG

            translation_openMVG : numpy array,
                                the translation vector for each view which used in OpenMVG

            camera_intrinsic_matrix_openMVG : numpy array shape(3x3)
                                            the camera intrinsic matrix which used in OpenMVG

            camera_intrinsic_dist_coefficients_openMVG : numpy array shape (1x5)
                                            the camera intrinisc distortion coefficients which used in OpenMVG
    """

    data = json.load(open(input_dir))
    # obtain the extrinsic data
    extrinsic_data = extract_element_from_json(data, ["extrinsics", ])
    # obtain the rotation matrix for each view
    rotation_openMVG = extract_element_from_json(extrinsic_data, ["value", "rotation"])[0]
    # make the value as the numpy array
    rotation_openMVG = np.array(rotation_openMVG)

    # obtain the center from OpenMVG json file
    center_openMVG = extract_element_from_json(extrinsic_data, ["value", "center"])[0]
    # turn the value into numpy array
    center_openMVG = np.array(center_openMVG)
    """
    the center_openMVG is not the translation vector, in order to turn it into translation vector,
    need to follow the reference website :
    https://openmvg.readthedocs.io/en/latest/openMVG/sfm/sfm/ (camera poses concept)
    https://github.com/openMVG/openMVG/blob/develop/src/openMVG/geometry/pose3.hpp (line 117 to 131)

    """
    # transfer center into translation vector
    translation_openMVG_list = []
    for i in range(len(rotation_openMVG)):
        translation_vector_openMVG = -np.matmul(rotation_openMVG[i] , center_openMVG[i])
        translation_openMVG_list.append(translation_vector_openMVG)

    translation_openMVG = np.array(translation_openMVG_list)

    # obtainthe intrinsic data
    intrinsic_data = extract_element_from_json(data, ["intrinsics"])
    value = extract_element_from_json(intrinsic_data, ["value"])
    # obtain the focal length from .json file
    focal_length_openMVG = np.array(extract_element_from_json(value, ["ptr_wrapper", "data", "focal_length"]))
    # obtain the principle points
    principal_point_openMVG = np.array(extract_element_from_json(value, ["ptr_wrapper", "data", "principal_point"])[0])
    # obtain the distortion coefficient value
    disto_k3 = extract_element_from_json(value,["ptr_wrapper","data", "disto_k3"])[0]

    def obtain_intrinsic_matrix(focal_length, principal_points):
        """
        this is a internal function to generate the intrinsic camera matrix by using the focal length and principal points

        Argument:
            focal_length_openMVG : numpy array
                            value which is obtained from sfm.json file in OpenMVG

            principal_points_openMVG : numpy array
                            value which is obtained from sfm.json file in OpenMVG
        
        Output:
            camera_intrinsic_matrix_openMVG: numpy array with shape(3x3)
                                             the camera intrinsic matrix of OpenMVG used
        """
        intrinsic_camera_matrix = np.ndarray(shape = (3,3))
        # make the focal length into camera intrinsic matrix
        intrinsic_camera_matrix[:,0][0] = focal_length
        intrinsic_camera_matrix[:,1][1] = focal_length
        # make the principal point into camera intrinsic matrix
        intrinsic_camera_matrix[:,2][0] = principal_points[:,0]
        intrinsic_camera_matrix[:,2][1] = principal_points[:,1]

        intrinsic_camera_matrix[2,:] = np.array([0.0, 0.0, 1])
        intrinsic_camera_matrix[0,1] = np.array([0])
        intrinsic_camera_matrix[1,0] = np.array([0])
        return intrinsic_camera_matrix
    """
    camera intrinsic matrix K

    K = [[fx, 0 , cx],
         [0, fy, cy],
         [0, 0, 1]]

    """
    camera_intrinsic_matrix_openMVG = obtain_intrinsic_matrix(focal_length = focal_length_openMVG,
                                                            principal_points = principal_point_openMVG)

    camera_intrinsic_dist_coefficients_openMVG = np.array([disto_k3[0][0],
                                                            disto_k3[0][1], 
                                                            0.0 , 
                                                            0.0 ,
                                                            disto_k3[0][2]])

    # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    if save_value_switch :
        np.save(os.path.join(output_dir, "camera_intrinsic_matrix_OpenMVG"), camera_intrinsic_matrix_openMVG)
        np.save(os.path.join(output_dir, "camera_distortion_coefficients_OpenMVG"), camera_intrinsic_dist_coefficients_openMVG)
        np.savez(os.path.join(output_dir, "rotation_matrix_OpenMVG"), rotation_openMVG)
        np.savez(os.path.join(output_dir, "center_OpenMVG"), center_openMVG)
        np.savez(os.path.join(output_dir, "translation_vector_OpenMVG"), translation_openMVG)

    print("values from OpenMVG result / file is found.")
    return rotation_openMVG, translation_openMVG, camera_intrinsic_matrix_openMVG, camera_intrinsic_dist_coefficients_openMVG


# %% Validation Projective matrix
def projective_matrix_validate(RGB_detected_object_points, 
                                RGB_camera_matrix_openCV, 
                                RGB_camera_dist_coeffs_openCV, 
                                IR_detected_object_points,
                                IR_camera_matrix_openCV,
                                IR_camera_dist_coeffs_openCV,
                                output_dir,
                                RGB_detected_object_points_dir = None, 
                                RGB_camera_matrix_openCV_dir = None, 
                                RGB_camera_dist_coeffs_openCV_dir = None, 
                                IR_detected_object_points_dir = None,
                                IR_camera_matrix_openCV_dir = None,
                                IR_camera_dist_coeffs_openCV_dir = None,  
                                save_value_switch = True,
                                file_read_switch = False):
    """
    This function is used to measure the error between the camera projective matrix and the projective matrix for each view
    In this validation, the validation will include 6 parts:
    Mean, Max, Min, Std, Coefficiient of Variance, and Norm 
    to validate if the camera projective matrix is reasonable or not

    the output will be a projective matrix error matrix

    Arguments :
            RGB_detected_object_points : numpy array 
                                        the RGB detected object points which generated during the RGB camera calibratiion

            RGB_camera_matrix_openCV : numpy array, shape (3x3)
                                        the RGB camera matrix which comes from the RGB camera calibration by using OpenCV
            
            RGB_camera_dist_coeffs_openCV : numpy array, shape (1x5) 
                                        the RGB camera distortion coefficients which comes from the RGB camera calibration by using OpenCV

            IR_detected_object_points : numpy array
                                        the IR detected object points which generated during the IR camera calibration
            
            IR_camera_matrix_openCV : numpy array, shape (3x3)
                                        the IR camera matrix which comes from the IR camera calibration by using OpenCV
            
            IR_camera_dist_coeffs_openCV : numpy array, shape (1x5)
                                        the IR camera distortion coefficients which comes from the IR camera calibration by using OpenCV

            save_value_switch : bool
                                the ON / OFF switch to save the information about the validation error result
                                the default is ON
    Output :
        Output_dir : str, 
                    the output working directory to save the dataframe
    """
    if file_read_switch: 
        # load npy, npz file
        # only .npz file need to add ["arr_0"] in order to read file.
        RGB_detected_points_read_path = np.load(RGB_detected_object_points_dir)["arr_0"]
        IR_detected_points_read_path = np.load(IR_detected_object_points_dir)["arr_0"]

        RGB_camera_matrix_read_path = np.load(RGB_camera_matrix_openCV_dir)
        RGB_camera_dist_read_path = np.load(RGB_camera_dist_coeffs_openCV_dir)

        IR_camera_matrix_read_path = np.load(IR_camera_matrix_openCV_dir)
        IR_camera_dist_read_path = np.load(IR_camera_dist_coeffs_openCV_dir)


        # make sure the shape is fit for the following steps
        RGB_detected_points_shape = RGB_detected_points_read_path.shape
        RGB_detected_points = np.array(RGB_detected_points_read_path).reshape(RGB_detected_points_shape[0]* RGB_detected_points_shape[1], RGB_detected_points_shape[3])

        IR_detected_points_shape = IR_detected_points_read_path.shape
        IR_detected_points = np.array(IR_detected_points_read_path).reshape(IR_detected_points_shape[0]* IR_detected_points_shape[1], IR_detected_points_shape[3])

        # undistort points
        detected_RGB_undist = cv2.undistortPoints(src = RGB_detected_points,
                                                cameraMatrix = RGB_camera_matrix_read_path,
                                                distCoeffs = RGB_camera_dist_read_path)

        detected_IR_undist = cv2.undistortPoints(src = IR_detected_points,
                                                cameraMatrix = IR_camera_matrix_read_path,
                                                distCoeffs = IR_camera_dist_read_path)

        projective_matrix_all_view = cv2.findHomography(srcPoints = detected_RGB_undist,
                                                dstPoints = detected_IR_undist,
                                                method = 0)[0]
        # obtain projective matrix for each view
        projective_matrix_each_view_list = []
        error_list = []
        norm_list = []

        RGB_detected_points_each_view = np.array(RGB_detected_points_read_path).reshape(RGB_detected_points_shape[0], RGB_detected_points_shape[1], RGB_detected_points_shape[3])
        IR_detected_points_each_view = np.array(IR_detected_points_read_path).reshape(IR_detected_points_shape[0], IR_detected_points_shape[1], IR_detected_points_shape[3])

        for i in range(len(RGB_detected_points_shape)):

            detected_RGB_undist_each_view  = cv2.undistortPoints(src = RGB_detected_points_each_view[i], #observed point coordinates, 2xN / Nx2 1-channel or 1xN / Nx1 2-channel
                                                    cameraMatrix = RGB_camera_matrix_openCV,
                                                    distCoeffs = RGB_camera_dist_coeffs_openCV)
            # IR undistorted points
            detected_IR_undist_each_view = cv2.undistortPoints(src = IR_detected_points_each_view[i],
                                                    cameraMatrix = IR_camera_matrix_openCV,
                                                    distCoeffs = IR_camera_dist_coeffs_openCV)

            projective_matrix_each_view = cv2.findHomography(srcPoints = detected_RGB_undist_each_view,
                                                            dstPoints = detected_IR_undist_each_view,
                                                            method = 0)[0]
            projective_matrix_each_view_list.append(projective_matrix_each_view)

            error = projective_matrix_each_view - projective_matrix_all_view
            error_list.append(error)
         
            norm = np.linalg.norm(error, axis = 0)
            error_list.append(error)
            norm_list.append(norm)

        mean_norm = np.mean(norm_list)
        max_norm = np.max(norm_list)
        min_norm = np.min(norm_list)
        std_norm = np.std(norm_list)
        # coefficient of variance = mean /std
        coefficient_var_norm = mean_norm / std_norm
        # save error values as dataframe
        df_error = pd.DataFrame({"norm" : [norm],
                                "mean_norm" : [mean_norm],
                                "max_norm" : [max_norm],
                                "min_norm" : [min_norm],
                                "std_norm" : [std_norm],
                                "coefficient_var_norm" : [coefficient_var_norm]})

    else : 
        
        # determine images view number
        n_images = len(RGB_detected_object_points)
        # determine detected points in each view
        # here use first image detected points as a reference number, because each view has same number of detected points
        n_points = len(RGB_detected_object_points[0])
        # get undistorted points
        # RGB undistorted points 
        RGB_detected_points = np.array(RGB_detected_object_points).reshape(n_images * n_points, 2)
        IR_detected_points = np.array(IR_detected_object_points).reshape(n_images * n_points, 2)

        detected_RGB_undist  = cv2.undistortPoints(src = RGB_detected_points, #observed point coordinates, 2xN / Nx2 1-channel or 1xN / Nx1 2-channel
                                                    cameraMatrix = RGB_camera_matrix_openCV,
                                                    distCoeffs = RGB_camera_dist_coeffs_openCV)
        # IR undistorted points
        detected_IR_undist = cv2.undistortPoints(src = IR_detected_points,
                                                cameraMatrix = IR_camera_matrix_openCV,
                                                distCoeffs = IR_camera_dist_coeffs_openCV)
        # obtain 1 projective matrix based on all views
        projective_matrix_all_view = cv2.findHomography(srcPoints = detected_RGB_undist,
                                                dstPoints = detected_IR_undist,
                                                method = 0)[0]
        # obtain projective matrix for each view
        projective_matrix_each_view_list = []
        error_list = []
        norm_list = []
        RGB_detected_points_each_view = np.array(RGB_detected_object_points).reshape(n_images, n_points, 2)
        IR_detected_points_each_view = np.array(IR_detected_object_points).reshape(n_images, n_points, 2)
        for i in range(n_images):
            detected_RGB_undist_each_view  = cv2.undistortPoints(src = RGB_detected_points_each_view[i], #observed point coordinates, 2xN / Nx2 1-channel or 1xN / Nx1 2-channel
                                                    cameraMatrix = RGB_camera_matrix_openCV,
                                                    distCoeffs = RGB_camera_dist_coeffs_openCV)
            # IR undistorted points
            detected_IR_undist_each_view = cv2.undistortPoints(src = IR_detected_points_each_view[i],
                                                    cameraMatrix = IR_camera_matrix_openCV,
                                                    distCoeffs = IR_camera_dist_coeffs_openCV)

            projective_matrix_each_view = cv2.findHomography(srcPoints = detected_RGB_undist_each_view,
                                                            dstPoints = detected_IR_undist_each_view,
                                                            method = 0)[0]
            projective_matrix_each_view_list.append(projective_matrix_each_view)

            error = projective_matrix_each_view - projective_matrix_all_view

            norm = np.linalg.norm(error, axis = 0)

            error_list.append(error)
            norm_list.append(norm)
        
        mean_norm = np.mean(norm_list)
        max_norm = np.max(norm_list)
        min_norm = np.min(norm_list)
        std_norm = np.std(norm_list)
        # coefficient of variance = mean /std
        coefficient_var_norm = mean_norm / std_norm
        # save error values as dataframe
        df_error = pd.DataFrame({"norm" : [norm],
                                "mean_norm" : [mean_norm],
                                "max_norm" : [max_norm],
                                "min_norm" : [min_norm],
                                "std_norm" : [std_norm],
                                "coefficient_var_norm" : [coefficient_var_norm]})

    # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    if save_value_switch :
        df_error.to_csv(path_or_buf = os.path.join(output_dir, "projective_matrix_error_matrix.csv"))
    
    
    return df_error
#%% Add distortion

def add_distortion(image_without_distortion, camera_distortion_matrix):
    """
    Here is the function that used to add distortion on images
    From [x', y'] to [x" , y"]

    Arguments :
            image_without_distortion : numpy array, shape (N x 2)
                                    the undistorted object points [x', y'] (in RGB plane or IR plane).
            
            camera_distortion_matrix : numpy array, shape (1 x 5)
                                    the camera distortion coefficients which generated by using OpenCV in camera calibration process.

    Output : 
            image_points_with_distortion : numpy array, shape (N x 2)
                                            the output for image_points_with_distortion will return the value which is [x" , y"]
    """
    # get x', y'
    image_without_distortion_x = np.array([])
    image_without_distortion_y = np.array([])

    image_without_distortion_x = image_without_distortion[:,0] 
    image_without_distortion_y = image_without_distortion[:,1] 
    # r^2 = x'^2 + y'^2
    r = np.sqrt(pow(image_without_distortion_x,2) + pow(image_without_distortion_y,2))
    """
    Methodology: 

        Add distortion :

        x" = x' * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) + 2 * p1 * x' * y' + p2 * (r^2 + 2 * x'^2)
        y" = y' * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) + p1 * (r^2 + 2 * y'^2) + 2 * p2 * x' * y'

        r^2 = x'^2 + y'^2

        Here : 
        [x' , y'] represent for the object points without distortion
        
    """
    image_with_distortion_x = image_without_distortion_x * (1 + camera_distortion_matrix[:,0]* pow(r,2) + camera_distortion_matrix[:,1]* pow(r,4) + camera_distortion_matrix[:,-1]* pow(r,6)) + 2 * camera_distortion_matrix[:,2] * image_without_distortion_x * image_without_distortion_y + camera_distortion_matrix[:,-2]*(pow(r,2) + 2 * pow(image_without_distortion_x,2))
    image_with_distortion_y = image_without_distortion_y * (1 + camera_distortion_matrix[:,0]* pow(r,2) + camera_distortion_matrix[:,1]* pow(r,4) + camera_distortion_matrix[:,-1]* pow(r,6)) + camera_distortion_matrix[:,2] * (pow(r,2) + 2 * pow(image_without_distortion_y,2)) + 2 * camera_distortion_matrix[:,-2] * image_without_distortion_x * image_without_distortion_y

    image_with_distortion = np.ndarray(shape = (len(image_without_distortion), 2))
    image_with_distortion[:, 0] = image_with_distortion_x
    image_with_distortion[:, 1] = image_with_distortion_y

    #image_with_distortion = np.vstack((image_with_distortion_x,image_with_distortion_y)).T

    return image_with_distortion
#%% Mapping RGB [x',y'] to IR [x', y']

# From RGB[x',y'] to IR[x',y']
# the shape of RGB_undistort_points is Nx2
# the shape of the projective_matrix_array is 3x3
def mapping_RGB_to_IR(RGB_undistort_points, projective_matrix):
    """
    Mapping the object points from RGB image [x',y'] to IR image [x',y']
    From RGB[x', y'] to IR[x', y']

    Arguments : 
            RGB_undistort_points : numpy array (shape = N x 2)
                                the object points without distortion in RGB[x',y'] image plane.
            
            projective_matrix : numpy array (shape = 3 x 3)
                                the projective matrix for mapping the points from RGB into IR, 
                                the projective matrix is generated in camera co-calibration by using OpenCV

    Output : 
            IR_undistorted_points : numpy array (shape = N x 2)
                                    the undistorted points in IR[x', y'] image plane.
    """
    # create homogenous coordinates
    N = len(RGB_undistort_points)
    image_points_2D_homogenous = np.ones(shape = (3, N))
    image_points_2D_homogenous[0:2,:] = RGB_undistort_points.T # RGB_undistort_points has a shape (N,2)
    # thermal_points_shape is 3xN
    # project_matrix_array shape is 3x3
    # the image_points_2D_homogenous is 3xN
    # the output shape of thermal_points_undistort_homogenous is 3 x N
    thermal_points_undistort_homgenous = projective_matrix @ image_points_2D_homogenous # A @ B is faster than np.matmul()
    # the shape of the thermal_image_points_undistort is Nx2
    thermal_image_points_undistort = np.empty(shape = (2,N)) # initialize an empty matrix
    thermal_image_points_undistort[0, :] = thermal_points_undistort_homgenous[0,:] / thermal_points_undistort_homgenous[2,:]
    thermal_image_points_undistort[1, :] = thermal_points_undistort_homgenous[1,:] / thermal_points_undistort_homgenous[2,:]
    return thermal_image_points_undistort.T

#%%
"""
Need to re-define / re-write the function
"""
# [u,v] = [fx * x" + cx, fy * y" + cy]
# therefore, the function of finding focal length and principal points will be define as
# find principal_point based in camera matrix
# the output will be 2 value, based on the intrinsic camera matrix
def find_principal_point(intrinsic_camera_matrix):
    """
    Obtain the principal point from the camera intrinsic matrix

    Arguments : 
            intrinsic_camera_matrix : numpy array (shape = 3 x 3)
    
    Output : 
            principle_points : numpy array (shape = 1 x 2)
                            the principle_points in the camera intrinsic matrix

    """
    principal_points = np.array([intrinsic_camera_matrix[0][2], intrinsic_camera_matrix[1][2]])
    # return is used to store the value in a function
    return(principal_points)
    
# find focal_length
def find_focal_length(intrinsic_camera_matrix):
    """
    Obtain the focal length from the camera intrinsic matrix

    Arguments :
            intrinsic_camera_matrix : numpy array (shape = 3 x 3)

    Output : 
            focal_length : numpy array (shape = 1 x 2)
                        the focal_length in the camera intrinsic matrix 
    """
    focal_length = np.array([intrinsic_camera_matrix[0][0], intrinsic_camera_matrix[1][1]])

    return(focal_length)

#from [x", y"] to [u,v]
def from_distortion_image_to_image_plane(image_with_distortion, camera_intrinsic_matrix):
    """
    Apply the [x" , y"] to [u, v]

    Arguments :

        image_with_distortion : numpy array (shape = N x 2)
                            [x" , y"] object points which after add distortion
    
        intrinsic_camera_matrix : numpy array (shape = 3 x 3)
                            the intrinsic_camera_matrix which is generated by OpenCV / OpenMVG
    Output :
        object_points_in_image_plane : numpy array (shape = N x 2)
                            the output shows the object points in the image plane [u,v]
    
    Methodolgy :

        u = fx * x" + cx
        v = fy * y" + cy 

        [fx , fy] represent for the focal length in x , y direction
        [cx , cy] represent for the principle points in x , y direction

        [x" , y"] represent for the object points with distortion
    """
    image_plane = np.ndarray(shape = (len(image_with_distortion), 2))
    image_plane[:, 0] = image_with_distortion[:, 0] * find_focal_length(camera_intrinsic_matrix)[0] + find_principal_point(camera_intrinsic_matrix)[0]
    image_plane[:, 1] = image_with_distortion[:, 1] * find_focal_length(camera_intrinsic_matrix)[1] + find_principal_point(camera_intrinsic_matrix)[1]

    return image_plane

# %% load ..ply file 
def load_ply_file(input_dir, output_dir, save_value_switch = False):
    """
    Here is the function to load .ply file which is generated by OpenMVG,
    in order to obtain all the points about the object points in 3D coordinate system. Shape (3x3)
    
    Argument :
            input_dir : str (Need to put the exactly file path)
                        the working directory for the .ply file
            
            save_value_switch : bool
                                the ON / OFF switch for saving the 3D object points as numpy array (.npz file)
                                the default is OFF
                        
    Output : 
            the output will be the object points in 3D coordinate system.
    
    Optional : 
            output_dir : str
                        the working directory for saving the object points
                        if the save_value_switch is ON, the output_dir has to be filled
            
    """
    # object points 3D coordinate information
    plydata = PlyData.read(input_dir)
    x = plydata['vertex'].data['x']
    y = plydata['vertex'].data['y']
    z = plydata['vertex'].data['z']
    # 3D object points
    object_points_3D = np.vstack((x,y,z)).T
    
     # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # save the result of the 3D object points 
    if save_value_switch :
        np.savez(os.path.join(output_dir, "3D_object_points"), object_points_3D)

    return object_points_3D
#%% From 3D objects to 2D RGB images [u,v] by using OpenCV
def from_3D_to_2D_IR(object_points_3D, 
                    rotation_matrix_openMVG,
                    translation_vector_openMVG,
                    camera_intrinsic_matrix_openMVG,
                    camera_distortion_coefficients_openMVG,
                    RGB_camera_matrix_openCV,
                    RGB_camera_dist_openCV,
                    projective_matrix,
                    IR_camera_matrix_openCV,
                    IR_camera_dist_openCV,
                    worst_points_uv, # [u,v]
                    output_dir,
                    object_points_3D_input_dir = None, 
                    rotation_matrix_openMVG_input_dir = None,
                    translation_vector_openMVG_input_dir = None,
                    camera_intrinsic_matrix_openMVG_input_dir = None,
                    camera_distortion_coefficients_openMVG_input_dir = None,
                    RGB_camera_matrix_openCV_input_dir = None,
                    RGB_camera_dist_openCV_input_dir = None,
                    projective_matrix_input_dir = None,
                    IR_camera_matrix_openCV_input_dir = None,
                    IR_camera_dist_openCV_input_dir = None,
                    path_use_switch = False,
                    save_value_switch = True):

    """
    From 3D object points [x,y,z] to 2D object points IR image [u,v]IR, all the input value comes from OpenMVG
    Also include the threshold as a fileter to only keep the high quality points as the output

    Arguments : 

        object_points_3D : numpy array (shape = N x 3)
                            the object points in 3D coordinate system, which is the densify point cloud generated by OpenMVS
        
        rotation_matrix_openMVG : numpy array (shape = N x 3 x 3)
                            rotation matrix for the object in different view, which is generated by OpenMVG
        
        translation_vector_openMVG : numpy array (shape = N x 1 x 3)
                            translation vector for the object in different view, which is generated by OpenMVG
                            In order to obtai the translation_vector, the "center" from OpenMVG should be calculated.
        
        camera_intrinsic_matrix_openMVG : numpy array (shape = 3 x 3)
                                        the intrinsic camera matrix which is generated by OpenMVG.

        camera_distortion_coefficents_openMVG : numpy array (shape = 1 x 5)
                                        the camera distortion coefficients which is generated by OpenMVG
        
        output_dir : str (Optional argument)
                    the output_dir for saving the result during the process.
                    IF save_value_switch is ON, the output_dir cannot be ignored

        save_value_switch : bool
                            the ON / OFF switch for saving the value which is generated during the process.
                            the default value is ON.
    
        path_use_switch : bool
                        the ON / OFF switch for using the local file instead of using the numpy array
                        the default value is OFF
            
    Optional: IF path_use_switch is ON

        object_points_3D_input_dir : str, (.ply or .npz file)
                            the input directory for the object points in 3D coordinates whiich is generated by OpenMVS

        rotation_matrix_openMVG_input_dir : str
                            the input directory for the rotation matrix for each view which generated by OpenMVG

        translation_vector_openMVG_input_dir : str 
                            the input directory for the translation vector for each view which calculated based on the center (generated by OpenMVG)

        camera_intrinsic_matrix_openMVG_input_dir : str 
                            the input directory for the camera intrinsic matrix for RGB camera which generated by OpenMVG

        camera_distortion_coefficients_openMVG_input_dir : str 
                            the input directory for the camera distortion coefficient for RGB camera which generated by OpenMVG

        RGB_camera_matrix_openCV_input_dir : str 
                            the input directory for the camera intrinsic matrix for RGB camera during the camera calibration which generated by OpenCV

        RGB_camera_dist_openCV_input_dir : str
                            the input directory for the camera distortion coefficient for RGB camera during the camera calibration which generated by OpenCV

        projective_matrix_input_dir : str
                            the input directory for the projective matrix which used to map the information from RGB image into IR image which generated by OpenCV 
                            during the camera CO- calibration

        IR_camera_matrix_openCV_input_dir : str
                            the input directory for the IR camera intrinsic matrix for IR camera during the camera calibratio, which is generated by OpenCV

        IR_camera_dist_openCV_input_dir : str
                            the input directory for the IR camera distortion coefficient for IR camera during the camera calibration, which is generated by OpenCV

    Output : 
            object_RGB_uv : numpy array
                            the object points mapped into RGB [u,v] plane

    """
    # process print
    print(" 3D object points coordinates to 2D IR image coordinate process is started .... ")

    # create list in order to save all the by-product value
    object_RGB_uv_list = []
    object_RGB_undist_list = []
    object_IR_without_dist_list = []
    object_IR_with_dist_list = []
    object_IR_uv_list = []

    if path_use_switch : 
        object_points_3D = np.load(object_points_3D_input_dir)["arr_0"]
        # load RGB camera information (OpenMVG)
        rotation_matrix_openMVG = np.load(rotation_matrix_openMVG_input_dir)["arr_0"]
        translation_vector_openMVG = np.load(translation_vector_openMVG_input_dir)["arr_0"]
        camera_intrinsic_matrix_openMVG = np.load(camera_intrinsic_matrix_openMVG_input_dir)
        camera_distortion_coefficients_openMVG = np.load(camera_distortion_coefficients_openMVG_input_dir)
        # load RGB camera information (OpenCV)
        RGB_camera_matrix_openCV = np.load(RGB_camera_matrix_openCV_input_dir)
        RGB_camera_dist_openCV = np.load(RGB_camera_dist_openCV_input_dir)
        # load IR camera information (OpenCV)
        IR_camera_matrix_openCV = np.load(IR_camera_matrix_openCV_input_dir)
        IR_camera_dist_openCV = np.load(IR_camera_dist_openCV_input_dir)
        # load projective matrix (OpenCV)
        projective_matrix = np.load(projective_matrix_input_dir)

        # number of total points for object points
        n_points = len(object_points_3D)
        # number of total view based on rotationn matrix from OpenMVG
        n_view = len(rotation_matrix_openMVG)

        # create for loop to obtain the high quality object points in IR image for each view.
        for i in tqdm(range(0, n_view)):
            # From 3D object points to 2D RGB image 
            # [x, y, z] to [u,v]RGB
            object_RGB_uv = cv2.projectPoints(objectPoints = object_points_3D,
                                            rvec = cv2.Rodrigues(rotation_matrix_openMVG[i])[0],
                                            tvec = translation_vector_openMVG[i].reshape(3,1),
                                            cameraMatrix = camera_intrinsic_matrix_openMVG,
                                            distCoeffs = camera_distortion_coefficients_openMVG)[0]
            
            object_RGB_uv_list.append(object_RGB_uv)
            # Remove distortion for 2D RGB image 
            # [u, v]RGB to [x', y']RGB
            """
            The general step for [u,v] to [x',y'] is 

            [u, v] ---> [x", y"] ----> [x',y']

            Here in this progress, by using cv2.undistortPoints() function can get the [x', y'] directly
            
            Reference website : 
            https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html

            """
            object_RGB_undist = cv2.undistortPoints(src = object_RGB_uv,
                                                    cameraMatrix = RGB_camera_matrix_openCV,
                                                    distCoeffs = RGB_camera_dist_openCV)

            object_RGB_undist_list.append(object_RGB_undist)
            # Mapping the object points from RGB[x', y'] to IR[x', y'] by using the projective matrix
            object_IR_without_dist = mapping_RGB_to_IR(RGB_undistort_points = object_RGB_undist.reshape(n_points, 2),
                                                        projective_matrix = projective_matrix)

            object_IR_without_dist_list.append(object_IR_without_dist)
            # add distortion on object IR[x', y']
            # IR[x', y'] to IR[x", y"]
            object_IR_with_dist = add_distortion(image_without_distortion = object_IR_without_dist,
                                                camera_distortion_matrix = IR_camera_dist_openCV)
            object_IR_with_dist_list.append(object_IR_with_dist)
            
            # threshold setup
            """
            In order to selected the high quality object points with high quality mapping result,
            the threshold should be setup.

            Reason for threshold setup: 
                In the mapping step, if the object points at the boundary of the RGB image, probably cannot be mapped in a correct place in IR image.
                
                The reason why this happen a lot is because of thermal camera (some of the thermal camera is kind close to fish eye camera)

                Therefore, In IR[x', y'] add distortion step could happend this issue.

            Methodology for threshold setup:

                1. Setup the boundary point as the threshold, and obtain the boundary points in [x", y"]IR image plane.

                2. Caculate the Norm of threshold point between [x', y']_norm and [x", y"]_norm.

                3. Compare the Norm between the object points and the threshold.

                4. If the Norm for the object points is bigger than the threshold, then ignore this object point.

            """
            # set up threshold point
            """

            the threshold setup is based on the worst points in the IR_image_uv, 
            because majority of the thermal camera has huge distortion (close to fisheye condition), 
            the distortion can cause the mapping result point far away from the correct position

            """
            IR_threshold_point = np.array([0, worst_points_uv[0]/2]).reshape(1, 2)
            # remove distortion from IR_threshold point
            IR_threshold_undist = cv2.undistortPoints(src = IR_threshold_point,
                                                        cameraMatrix = IR_camera_matrix_openCV,
                                                        distCoeffs = IR_camera_dist_openCV)
            # add distortion for the threshold
            IR_threshold_dist = add_distortion(image_without_distortion = IR_threshold_undist.reshape(1,2),
                                            camera_distortion_matrix = IR_camera_dist_openCV)
            # calculate the threshold norm
            threshold_norm = np.linalg.norm(IR_threshold_dist - IR_threshold_undist)

            # get norm for the object points after distortion step
            object_points_norm = np.linalg.norm(object_IR_with_dist - object_IR_without_dist, axis = 1)
            # set up flag to find the best points
            a = 0 <= threshold_norm
            b  = threshold_norm >= object_points_norm
            # use bool value method
            flag_IR = a & b
            # set up an empty numpy array with N x 2 shape
            object_IR_uv = np.empty(shape = (n_points, 2))
            # use NA to fill all the value for object_IR_uv
            object_IR_uv[:, :] = np.nan
            # use bool value to select the points which statisfy the flag condition
            IR_object_points_with_dist = object_IR_with_dist_list[i][flag_IR]
            # apply [x", y"] to [x', y'] on statisfied object points
            object_IR_uv[flag_IR] = from_distortion_image_to_image_plane(image_with_distortion = IR_object_points_with_dist,
                                                                        camera_intrinsic_matrix = IR_camera_matrix_openCV)
            object_IR_uv_list.append(object_IR_uv)

    else :
        # number of total points for object points
        n_points = len(object_points_3D)
        # number of total view based on rotationn matrix from OpenMVG
        n_view = len(rotation_matrix_openMVG)
        # create for loop to obtain the high quality object points in IR image for each view.
        for i in tqdm(range(0, n_view)):
            # From 3D object points to 2D RGB image 
            # [x, y, z] to [u,v]RGB
            object_RGB_uv = cv2.projectPoints(objectPoints = object_points_3D,
                                            rvec = cv2.Rodrigues(rotation_matrix_openMVG[i])[0],
                                            tvec = translation_vector_openMVG[i].reshape(3,1),
                                            cameraMatrix = camera_intrinsic_matrix_openMVG,
                                            distCoeffs = camera_distortion_coefficients_openMVG)[0]
            object_RGB_uv_list.append(object_RGB_uv)
            # Remove distortion for 2D RGB image 
            # [u, v]RGB to [x', y']RGB
            """
            The general step for [u,v] to [x',y'] is 

            [u, v] ---> [x", y"] ----> [x',y']

            Here in this progress, by using cv2.undistortPoints() function can get the [x', y'] directly
            
            Reference website : 
            https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html

            """
            object_RGB_undist = cv2.undistortPoints(src = object_RGB_uv,
                                                    cameraMatrix = RGB_camera_matrix_openCV,
                                                    distCoeffs = RGB_camera_dist_openCV)
            object_RGB_undist_list.append(object_RGB_undist)
            # Mapping the object points from RGB[x', y'] to IR[x', y'] by using the projective matrix
            object_IR_without_dist = mapping_RGB_to_IR(RGB_undistort_points = object_RGB_undist.reshape(n_points, 2),
                                                        projective_matrix = projective_matrix)
            object_IR_without_dist_list.append(object_IR_without_dist)
            # add distortion on object IR[x', y']
            # IR[x', y'] to IR[x", y"]
            object_IR_with_dist = add_distortion(image_without_distortion = object_IR_without_dist,
                                                camera_distortion_matrix = IR_camera_dist_openCV)
            object_IR_with_dist_list.append(object_IR_with_dist)
            
            # threshold setup
            """
            In order to selected the high quality object points with high quality mapping result,
            the threshold should be setup.

            Reason for threshold setup: 
                In the mapping step, if the object points at the boundary of the RGB image, probably cannot be mapped in a correct place in IR image.
                
                The reason why this happen a lot is because of thermal camera (some of the thermal camera is kind close to fish eye camera)

                Therefore, In IR[x', y'] add distortion step could happend this issue.

            Methodology for threshold setup:

                1. Setup the boundary point as the threshold, and obtain the boundary points in [x", y"]IR image plane.

                2. Caculate the Norm of threshold point between [x', y']_norm and [x", y"]_norm.

                3. Compare the Norm between the object points and the threshold.

                4. If the Norm for the object points is bigger than the threshold, then ignore this object point.

            """
            # set up threshold point
            IR_threshold_point = np.array([0, worst_points_uv[0] / 2]).reshape(1,2)
            # remove distortion from IR_threshold point
            IR_threshold_undist = cv2.undistortPoints(src = IR_threshold_point,
                                                        cameraMatrix = IR_camera_matrix_openCV,
                                                        distCoeffs = IR_camera_dist_openCV)
            # add distortion for the threshold
            IR_threshold_dist = add_distortion(image_without_distortion = IR_threshold_undist.reshape(1,2),
                                            camera_distortion_matrix = IR_camera_dist_openCV)
            # calculate the threshold norm
            threshold_norm = np.linalg.norm(IR_threshold_dist - IR_threshold_undist)

            # get norm for the object points after distortion step
            object_points_norm = np.linalg.norm(object_IR_with_dist - object_IR_without_dist, axis = 1)
            # set up flag to find the best points
            a = 0 <= threshold_norm
            b  = threshold_norm >= object_points_norm
            # use bool value method
            flag_IR = a & b
            # set up an empty numpy array with N x 2 shape
            object_IR_uv = np.empty(shape = (n_points, 2))
            # use NA to fill all the value for object_IR_uv
            object_IR_uv[:, :] = np.nan
            # use bool value to select the points which statisfy the flag condition
            IR_object_points_with_dist = object_IR_with_dist_list[i][flag_IR]
            # apply [x", y"] to [x', y'] on statisfied object points
            object_IR_uv[flag_IR] = from_distortion_image_to_image_plane(image_with_distortion = IR_object_points_with_dist,
                                                                        camera_intrinsic_matrix = IR_camera_matrix_openCV)
            object_IR_uv_list.append(object_IR_uv)

    print("threshold_norm :", threshold_norm)
    # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # check ON /OFF switch
    if save_value_switch :
        # save high quality points
        np.savez(os.path.join(output_dir, "object_RGB_uv_multiview"), object_RGB_uv_list)
        np.savez(os.path.join(output_dir, "object_RGB_undist_multiview"), object_RGB_undist_list)
        np.savez(os.path.join(output_dir, "object_IR_without_dist_multiview"), object_IR_without_dist_list)
        np.savez(os.path.join(output_dir, "object_IR_with_dist_multiview"),object_IR_with_dist_list)
        np.savez(os.path.join(output_dir, "object_IR_uv_multiview"), object_IR_uv_list)

    # process report print
    print(" 3D object points coordinates to 2D IR image coordinate process is finished. ")

    return object_IR_uv_list

# %% obtain object gray image
def obtain_IR_image_gray(object_IR_image_input_dir, 
                        object_IR_prefix,
                        object_IR_image_format,
                        SCALE,
                        output_dir,
                        save_value_switch = True):
    """
    This function is used to obtain the IR image gray as a numpy array

    Arguments :
        object_IR_image_input_dir : str 
                        the input directory for the object IR images

        object_IR_prefix : str 
                        the prefix name for the object IR images

        object_IR_image_format : str
                        the format name for the object IR images 
                        (for example : JPG, JPEG, TIFF)

        SCALE : int or float
                the int or float to scale the IR images

        output_dir : str
                    the output directory for the object IR images gray

        save_value_switch : bool
                    the ON / OFF switch for saving the results value
                    the default is ON
    
    Output : 
        object_IR_gray_image : numpy array (shape = M x N)
                    the object IR image into gray scale image (change the IR image channel into 1)
    
    """
    # glob.glob() : obtain the path 
    object_IR_images = glob.glob(object_IR_image_input_dir + '/' + object_IR_prefix + '*.' + object_IR_image_format)
    object_IR_images = natsorted(object_IR_images)
    # the thermal images has to scale as same size as RGB image
    # read image and resize image
    SCALE = 12.6 # set up the number of scale needed
    object_IR_image_list = []
    object_IR_gray_image_list = []

    for fname in object_IR_images:
        # print("read")
        # cv2.imread() : read image 
        object_IR_img = cv2.imread(fname)
        # find the width and height for IR image to do scale 
        width = int(object_IR_img .shape[1] * SCALE)
        height = int(object_IR_img .shape[0] * SCALE)
        dim = (width, height)

        # cv2.resize() : resize the image size 
        object_IR_img  = cv2.resize(src = object_IR_img , 
                        dsize = dim, 
                        interpolation = cv2.INTER_LANCZOS4) # has to use cv2.INTER_LANCZOS4 to do interpolation for IR image
                        # the result for the scaled image will be easy to detected by using blob detector
        object_IR_image_list.append(object_IR_img)

        object_IR_gray_image = cv2.cvtColor(object_IR_img , cv2.COLOR_BGR2GRAY)

        object_IR_gray_image = np.array(object_IR_gray_image).astype(float)
        object_IR_gray_image_list.append(object_IR_gray_image)

    # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
    if save_value_switch :
        # save value 
        np.savez(os.path.join(output_dir, "object_IR_gray_image"), object_IR_gray_image_list)

    return object_IR_gray_image_list


# %% Mapping the temperature to high quality points
def temperature_mapping(object_points_IR_uv,
                        IR_gray_image_array,
                        IR_image_shape, #[u,v]
                        output_dir,
                        save_value_switch = True,
                        path_use_switch = False,
                        object_points_IR_uv_input_dir = None,
                        IR_gray_image_array_input_dir = None):

    """
    This function is used to find the temperature information for the high quality object points in the [u,v]IR plane

    Arguments : 

            object_points_IR_uv : numpy array (shape = n_view , n_total_points , 2)
                                the object points in IR[u,v] plane
            
            IR_gray_image_array : numpy array (shape = n_view , u , v)
                                the numpy array for the IR gray scale image

            IR_image_shape : numpy array (shape = M x N)
                            the shape for the SCALED IR image

            save_value_switch : bool
                                the ON / OFF switch for saving the result output
                                the default is ON

            path_use_switch : bool
                            the ON / OFF switch for using the work directory path instead of numpy array as input
                            the default is OFF
    Output : 

            temperature_pixel_list : list / numpy array
                                    the result for mapping the temperature information to the object points.
    
    Optional : if path_use_switch is ON
            
            object_points_IR_uv_input_dir : str
                                    the working directory for saved object points in IR[u,v]

            IR_gray_image_array_input_dir : str
                                    the working directory for saved IR gray scale image array
    
    """
    # process print
    print(" Temperature mapping is started .....")

    temperature_pixel_list = []
    if path_use_switch : 
        # load numpy array by using path
        object_points_IR_uv = np.load(object_points_IR_uv_input_dir)["arr_0"]
        IR_gray_image_array = np.load(IR_gray_image_array_input_dir)["arr_0"]
        n_view = len(object_points_IR_uv)
        n_total_points = len(object_points_IR_uv[0])

    else :
        n_view = len(object_points_IR_uv)
        n_total_points = len(object_points_IR_uv[0])

    for i in tqdm(range(0, n_view)):
        # thermal bool
        c = np.array(np.logical_and(0 <= object_points_IR_uv[i][:,0], object_points_IR_uv[i][:,0] <= IR_image_shape[0])) #u
        d = np.array(np.logical_and(0 <= object_points_IR_uv[i][:,1], object_points_IR_uv[i][:,1] <= IR_image_shape[1])) #v
        flag_bool = c & d # points in IR image is True
        # generate an empty numpy array with Nan value
        temperature_pixel = np.empty(shape = n_total_points, dtype = np.float64)
        temperature_pixel[:] = np.nan
        # pixel position
        pixel_position_bool = object_points_IR_uv[i].astype(int)[flag_bool]
            # use array()[x,y] to find the pixel number
            #i.e. u = X, v = Y (as if image shown using imshow(), 
            # the origin of the coordinate frame for consecutive plots is set to the image coordinate frame origin which is the top left corner).
            # x is the columns
            # y is the rows
            # IR[v,u] = pixel
        temperature_pixel[flag_bool] = IR_gray_image_array[i][pixel_position_bool[:,1] , pixel_position_bool[:, 0]]

        temperature_pixel_list.append(temperature_pixel)

    # create the output directory if not present
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
    if save_value_switch :
        # save value 
        np.savez(os.path.join(output_dir, "temperature_on_pixel_multiview"), temperature_pixel_list)
    # process print
    print(" Temperature mapping is finished. ")

    return temperature_pixel_list
#%% draw colors for the object points based on the temperature

def cstm_inferno(x, vmin, vmax):
    """
    Define a fit color for the object points based on the inferno color map 
    Scale the temperature into 0~255 in color map

    Arguments :
            x : numpy array
                mean / median temperature value of the same point in each view.

            vmin : 0
                the minmium value of the color map
            
            vmax : 255
                the maximum value of the color map

    Output :
            SudoRGB : numpy array 
                the SudoRGB value for the object points (another color for 3D thermal information)
    """
    # referece website
    # https://www.delftstack.com/howto/matplotlib/convert-a-numpy-array-to-pil-image-python/
    # normalize np.array() first
    return plt.cm.inferno((np.clip(x,vmin,vmax)-vmin)/(vmax-vmin))
#%% Radiometric correction
def radiometric_correction(pixel_temperature):
    """
    This function is applying the radiometric correction into the object 3D temperture result.
    The goal is making sure that the temperature in different views are in same temperature range.
    Also, in this function removed the outliers which generated during the Temperature Mapping step

    Argument :
            pixel_temperature : list / numpy array (shape = n_view, n_points, 1)
                            the temperature for object points in each pixel in different view
    
    Output : 
            new_median_temperature : numpy array (shape = n_view, n_points , 1)
                            new temperature for object points in each pixel in different view (after radiometric correction)
    """
    # process print
    print("Radiometric correction is started ....")

    new_temperature_list = []
    beta_list = []
    new_paired_points_list = []
    
    n_view = len(pixel_temperature)
    """
    All the radiometric correction is based on the first image.
    Find the linear regression between the temperature value in the first image and anyother images.
    """
    for i in tqdm(range(0, n_view)):
        # find nan points in view 0
        e = ~np.isnan(pixel_temperature[0])
        # find nan points in other view
        f = ~np.isnan(pixel_temperature[i])
        flag_paired = e & f # only paired the TRUE value
        # remove outliers
        temperature_in_2_views = np.array([pixel_temperature[0], pixel_temperature[i]]).T
        paired_points = temperature_in_2_views[flag_paired]
        # for the same object point 
        # if the temperature error is bigger than 70 
        # then remove the temperature value for that point
        new_paired_points = np.delete(paired_points, np.where(np.absolute(paired_points[:,1] - paired_points[:,0]) >= 70), axis = 0)
        new_paired_points_list.append(new_paired_points)
        # find linear regression between paired points
        y = new_paired_points[:, 0]
        x = np.ones(shape = (len(new_paired_points), 2))
        x[:, 1] = new_paired_points[:, 1]
        # beta = [b, k]
        beta = np.matmul(np.matmul(np.linalg.inv((np.matmul(np.transpose(x),x))),np.transpose(x)),y)
        beta_list.append(beta)
        slope = beta[1]
        intercept = beta[0]
        new_temperature = slope * pixel_temperature[i] + intercept
        """
        This function will only return the value of the temperature in each view for the object.
        It will has N_points x N_view as the shape of the result 
        
        return value format : list
        """
        new_temperature_list.append(new_temperature)
    # process print
    print("Radiometric correction is finished. ")
    return new_temperature_list
#%% Map color
def color_mapping_3d(input_dir,
                    temperature_value):
    """
    this function is used to replace the RGB color in .ply file into sudoRGB color

    Arguments : 
            input_dir : str,
                    the work directory for saving the .ply file which generated by OpenMVS
            
            temperature_value : numpy array (shape = n_points, 1)
                    the median / mean temperature value which is generated after radiometric
    """
    # process print
    print("Color mapping is started ....")
    # create 3 channel sudoRGB 
    sudoRGB = cstm_inferno(x = temperature_value,
                            vmin = 0,
                            vmax = 255)[:, 0:3]

    sudoRGB = np.uint8(sudoRGB * 255)[:, 0:3]
    # read object .ply file which generated by OpenMVS
    plydata = PlyData.read(input_dir)
    # copy the original data to avoid mess up the orginal data
    plydata_IR = copy.deepcopy(plydata)
    # Replace the RGB channel into the sudoRGB channel
    plydata_IR['vertex'].data['red'] = sudoRGB[:,0]
    plydata_IR['vertex'].data['green'] = sudoRGB[:,1]
    plydata_IR['vertex'].data['blue'] = sudoRGB[:,2]

    # process print
    print("Color mapping is finished. Thermal 3D object is generated. ")
    return plydata_IR

# %%
