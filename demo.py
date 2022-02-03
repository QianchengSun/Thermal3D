"""
This is a demo code to show how to apply the therm3d package for mapping the thermal information into 
3D reconstruction .ply file
"""
#%%
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
import os
import sys 
thermal3D_path = os.path.join(__file__)
sys.path.append(thermal3D_path)
import therm3d

def main():
    root = "/Volumes/GoogleDrive/"
    # set up path for the RGB camera calibration data
    RGB_image_folder = os.path.join(root,'Shared drives/therm3d/trash_can_example/8_5_afternoon/RGB_8_5_outdoor_calibration')
    RGB_prefix_name = "RGB_Outdoor_Calibration"
    RGB_image_format_name = "JPG"
    # set up path for IR camera calibration
    IR_image_folder = os.path.join(root,"Shared drives/therm3d/trash_can_example/8_5_afternoon/IR_0805_outdoor")
    IR_prefix_name = "IR_Outdoor_Calibration"
    IR_image_format_name = "JPG"
    # set up path for object IR image
    IR_object_image_folder = os.path.join(root,"Shared drives/therm3d/trash_can_example/8_5_afternoon/IR_Trash_can_0805")
    IR_object_prefix_name = "IR_trash_can"
    IR_object_format_name = "JPG"
    # set up path for the OpenMVG result
    sfm_json_input_dir = os.path.join(root,"Shared drives/therm3d/trash_can_example/trashcan_result_08_05/convert_sfm/sfm_data-2.json")
    object_ply_path = os.path.join(root,"Shared drives/therm3d/trash_can_example/trashcan_result_08_05/MVS_result/scene_dense.ply")
    # set up the path for output result
    demo_output_dir = r"/Users/qianchengsun/PhD/python_code_practice/demo"
    """
    Here SCALE is applied on the IR image, because the resolution of IR images are too small.
    Therefore, the IR calibration result should be different from it should be (factory setting).
    """
    # set up SCALE
    SCALE = 12.6

    # RGB camera calibration 
    RGB_camera_mtx, RGB_camera_dist, RGB_camera_img_points = therm3d.RGB_camera_calibration(
                                                                                        RGB_image_folder = RGB_image_folder,
                                                                                        RGB_prefix = RGB_prefix_name,
                                                                                        RGB_image_format = RGB_image_format_name,
                                                                                        RGB_output_dir = demo_output_dir,
                                                                                        save_value_switch = False)
    # IR camera calibration 
    IR_camera_matrix, IR_camera_dist, IR_detected_points = therm3d.IR_camera_calibration(IR_image_folder = IR_image_folder,
                                                                                        IR_prefix = IR_prefix_name,
                                                                                        IR_image_format = IR_image_format_name,
                                                                                        SCALE = SCALE,
                                                                                        IR_output_dir = demo_output_dir,
                                                                                        save_value_switch = False)
    """
    Here a comparsion of RGB /IR camera calibration result.
    In order to find the projective matrix between RGB camera and IR camera.
    The RGB / IR camera calibration result should be paired with each other.
    """
    if len(RGB_camera_img_points) == len(IR_detected_points):
        print("RGB calibrate images and IR calibrate images are paired.")
    elif len(RGB_camera_img_points) > len(IR_detected_points):
        print("Detected IR keypoints are less than RGB keypoints, please re-do the camera calibration! ")
    else:
        print("Detected RGB keypoints are less than IR keypoints, please re-do the camera calibration! ")

    # find projective matrix
    """
    There will only 1 projective matrix be generated by using paired RGB & IR detected points.
    """
    projective_matrix = therm3d.obtain_projective_matrix(RGB_detected_points = RGB_camera_img_points,
                                                        IR_detected_points = IR_detected_points,
                                                        RGB_camera_matrix = RGB_camera_mtx,
                                                        IR_camera_matrix = IR_camera_matrix,
                                                        RGB_camera_dist = RGB_camera_dist,
                                                        IR_camera_dist = IR_camera_dist,
                                                        output_dir = demo_output_dir,
                                                        read_dir_switch = False,
                                                        save_value_switch = False)
    # obtain the information from OpenMVG sfm.json file
    rotation_matrix_OpenMVG, translation_vector_OpenMVG, camera_matrix_OpenMVG, camera_dist_OpenMVG = therm3d.obtain_information_openMVG(input_dir = sfm_json_input_dir,
                                                                                            output_dir = demo_output_dir ,
                                                                                            save_value_switch= False)
    # load object .ply data
    object_points_3D = therm3d.load_ply_file(input_dir= object_ply_path,
                                            output_dir = demo_output_dir ,
                                            save_value_switch = False)

    # from 3D object points [x, y, z] to 2D IR [u,v]
    object_IR_uv = therm3d.from_3D_to_2D_IR(object_points_3D = object_points_3D,
                                            rotation_matrix_openMVG = rotation_matrix_OpenMVG,
                                            translation_vector_openMVG = translation_vector_OpenMVG,
                                            camera_intrinsic_matrix_openMVG = camera_matrix_OpenMVG,
                                            camera_distortion_coefficients_openMVG = camera_dist_OpenMVG,
                                            RGB_camera_matrix_openCV = RGB_camera_mtx,
                                            RGB_camera_dist_openCV = RGB_camera_dist,
                                            IR_camera_matrix_openCV = IR_camera_matrix,
                                            IR_camera_dist_openCV = IR_camera_dist,
                                            projective_matrix = projective_matrix,
                                            output_dir = demo_output_dir ,
                                            worst_points_uv = (4032,3024), # [u,v]
                                            path_use_switch= False,
                                            save_value_switch= False)
    # compare OpenCV camera calibration result with OpenMVG result
    print(camera_matrix_OpenMVG)
    print(RGB_camera_mtx)
    # obtain the object thermal image as gray scale
    IR_image_gray_array = therm3d.obtain_IR_image_gray(object_IR_image_input_dir = IR_object_image_folder,
                                                    object_IR_prefix = IR_object_prefix_name,
                                                    object_IR_image_format = IR_object_format_name,
                                                    SCALE = SCALE,
                                                    output_dir = demo_output_dir ,
                                                    save_value_switch = False)
    # otain the temperature information for each points in different view
    object_temperature = therm3d.temperature_mapping(object_points_IR_uv = object_IR_uv,
                                                    IR_gray_image_array = IR_image_gray_array,
                                                    IR_image_shape = (4032, 3024), #[u,v]
                                                    output_dir = demo_output_dir ,
                                                    save_value_switch = False,
                                                    path_use_switch = False)

    #  print the histogram plot before radiometric
    """
    The goal is plot the histogram plot about the temperature on 3D objects,
    based on the histogram plot to validate if the temperature of the object points need radiometric correction or not.

    """
    # In order to avoid the warning message about All-nan
    # If a points in all views are NAN value, then this point should be removed
    flag = ~np.isnan(object_temperature) # set up flag, True mean has value

    object_temperature_array = np.array(object_temperature)

    object_temperature_value_list = []
    """
    Here can be improved to avoid for loop applying on each points

    """
    for i in range(0, len(object_temperature[0])): # use single points 
        """
        Here, in order to select the object points which has value in each view.
        Using np.count_nonzero > 0 as a threshold, to make sure the object points temperature in each view at least has 1 value.
        Also, this step can avoid the warning message by applying np.nanmin(), np.nanmin()
        """
        if np.count_nonzero(flag[:,i]) > 0:
            object_temperature_value = object_temperature_array[:,i]
            object_temperature_value_list.append(object_temperature_value)                                           
    # histogram plot without the radiometric correction
    T_min = np.nanmin(object_temperature_value_list, axis = 1) # set up axis = 1, is because here the shape of 
    T_max = np.nanmax(object_temperature_value_list, axis = 1)
    T_range = T_max - T_min
    plt.hist(T_range)           

    #  Radiometric correction
    """
    Here is the step to do the radiometric correction, 
    Based on pervious histogram plot about the error range for single object points in thermal 3D modeling.
    In order to reduce the error range for each object points in thermal 3D modeling
    The radiometric has been applied on the mapping result
    """
    new_object_temperature = therm3d.radiometric_correction(object_temperature)

    # set up a flag as a filter to select all the points without NA values 
    # in order to avoid warnings
    flag_radiometric_correction = ~np.isnan(new_object_temperature)
    # turn list into numpy array
    new_object_temperature_array = np.array(new_object_temperature)
    # create a list to append all new filtered points
    new_object_temperature_value_list = []
    for i in range(0, len(new_object_temperature[0])):
        if np.count_nonzero(flag_radiometric_correction[:,i]) > 0 :
            new_object_temperature_value = new_object_temperature_array[:, i]
            new_object_temperature_value_list.append(new_object_temperature_value)

    # histogram after the radiometric correction
    T_min_radiometric = np.nanmin(new_object_temperature_value_list, axis = 1)
    T_max_radiometric = np.nanmax(new_object_temperature_value_list, axis = 1)
    T_range_radiometric = T_max_radiometric - T_min_radiometric
    plt.hist(T_range_radiometric)  
    plt.title("Temperature Error Range")
    """
    Here use np.nanmedian() to obtain the mean temperature of the object points in different thermal view

    Discussion: 
    Using Median can avoid the object points temperature being over estimated. 
    Because the maxmium, and minmium of the object points temperature in different view could have big difference, 
    since the upper and lower bound temperature of thermal image could be change a lot.

    """
    new_mediam_temperature = np.nanmedian(new_object_temperature , axis = 0)
    # generate thermal 3D object
    data_3d = therm3d.color_mapping_3d(input_dir = object_ply_path,
                                    temperature_value = new_mediam_temperature)
    PlyData.write(data_3d, os.path.join(demo_output_dir, "demo.ply"))

#%% Apply main function
if __name__ == '__main__':
    main()



# %%
