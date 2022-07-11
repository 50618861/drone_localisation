"""
Image Stitching Algrithm


Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Stitch LeadingEdge of Blade A without refinement
    python3 image_stitching.py --dataset '/home/temp-dhalion-linux/yixiang/Dataset/gossip-assets-files_new.xlsx' --output /home/uob/drone_localization/output --beg 3 --end 14

    # Stitch LeadingEdge of Blade A with refinement
    python3 image_stitching.py --dataset '/home/temp-dhalion-linux/yixiang/Dataset/gossip-assets-files_new.xlsx' --output /home/uob/drone_localization/output --beg 3 --end 14 --refinement True --visulization True --weight /home/temp-dhalion-linux/yixiang/localisation/3rd-party/maks_rcnn_blade_0046.h5 --point_cloud_model_path /home/temp-dhalion-linux/yixiang/Dataset/advanced_model/manual_ritual_turbine_points_blade.ply

"""

# import open3d as o3d
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import sys
import cv2 as cv

import refinement
import math

import matplotlib.pyplot as plt

class Localization():
    def __init__(self):        
        self.intrinsic_matric_K = np.array([
                                        [1787.69, 0, 320],
                                        [0, 2008.34, 240],
                                        [0, 0, 1]
                                    ], dtype='f') # for image size (640*480)

        self.image_width = 640
        self.image_heigh = 480

        # self.centre_2_blade = 6.3 # The distance from coordinate centre to blade surface
        # self.centre_2_TE = 3.8
        self.centre_2_blade = 0 # The distance from coordinate centre to blade surface
        self.centre_2_TE = 0

    def resize (self, img, width, height):
        dim = (width, height)
        
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        return img

    def get_cam_pos(self, data,num):   
        return [data[num][8],data[num][7],data[num][6],data[num][9]]

    def get_img(self, data,num):
        # print(os.path.join(image_data_path, data[num][4]))
        img = cv.imread(os.path.join(image_data_path, data[num][4]))
        img = self.resize(img, self.image_width, self.image_heigh)
        return img

    def get_cam_rot(self,data,num):   
        return [data[num][10],data[num][11],data[num][12]]

    def world_2_camera(self, drone_tran,drone_rot):

        t = np.array([drone_tran[1],-drone_tran[3],drone_tran[0]])
        r = R.from_euler("xzy", drone_rot, degrees=True)

        rotation_matrix = r.as_matrix()
        rotation_matrix = np.linalg.inv(rotation_matrix)

        trans_t = rotation_matrix.dot(t.T)

        return -trans_t

    def correction_with_rotation(self, drone_rot,x,y,z,blade_position,yaw):


        if blade_position == "A":

            if yaw < 180:
                x = x - z*math.tan(math.radians(drone_rot[2]))
            else:
                x = x + z*math.tan(math.radians(drone_rot[2]))

            y = y - z*math.tan(math.radians(drone_rot[0]))

        if blade_position == "B" or blade_position == "C":

            if yaw < 180:
                x = x + z*math.tan(math.radians(drone_rot[2]))
            else:
                x = x - z*math.tan(math.radians(drone_rot[2]))

            y = y + z*math.tan(math.radians(drone_rot[0]))

        return x,y

    def calculate_img_pos(self, drone_tran,drone_rot,distance , blade_side ,blade_position,yaw,image_location):

        x = -drone_tran[0] * 1000 
        y = drone_tran[1] * 1000

        # if blade_side == "LeadingEdge":
        #     z = (drone_tran[2] - self.centre_2_blade) * 1000
        # elif blade_side == "TrailingEdge":
        #     z = (drone_tran[2] + self.centre_2_TE) * 1000

        # else:
        #     z = distance * 1000
        #     x,y = self.correction_with_rotation(drone_rot,x,y,z,blade_position,yaw)

        z = distance * 1000
        x,y = self.correction_with_rotation(drone_rot,x,y,z,blade_position,yaw)

        move_x = int((x/z) * self.intrinsic_matric_K[0,0])
        move_y = int((y/z) * self.intrinsic_matric_K[1,1])
    
        print(move_x,move_y)
        image_location.append([move_x,move_y])
       

    def stitching(self, init_img,img, move_x,move_y,pre_image_location):
        if move_x < -self.image_width:
            move_x = -self.image_width
        elif move_x > self.image_width:
            move_x = self.image_width
            
        if move_y < -self.image_heigh:
            move_y = -self.image_heigh
        elif move_y > self.image_heigh:
            move_y = self.image_heigh

        cur_image_location_y = pre_image_location[1] + move_y if pre_image_location[1] + move_y < 0 else 0
        cur_image_location_x = pre_image_location[0] + move_x if pre_image_location[0] + move_x > 0 else 0
        
        cur_image_location = [cur_image_location_x,cur_image_location_y]

        if move_y > 0 :
            
            img_width = int(max(init_img.shape[1] + abs(move_x), init_img.shape[1]))
            img_height = int(max(cur_image_location[1] + init_img.shape[0] +  abs(move_y), init_img.shape[0]))

            blank_image = np.zeros((img_height,img_width,3), np.uint8)

            if move_x > 0:
            
                blank_image[img_height - init_img.shape[0]:img_height,0:init_img.shape[1]] = init_img
                blank_image[abs(cur_image_location[1]):abs(cur_image_location[1]) + img.shape[0], cur_image_location[0]:cur_image_location[0] + img.shape[1]] = img
            else:

                blank_image[img_height - init_img.shape[0]:img_height,img_width - init_img.shape[1]:img_width] = init_img
                blank_image[abs(cur_image_location[1]):abs(cur_image_location[1]) + img.shape[0], cur_image_location[0]:cur_image_location[0] + img.shape[1]] = img

        elif move_y <= 0:
            img_width = int(max(init_img.shape[1] + abs(move_x), init_img.shape[1]))
            img_height = int(max(-cur_image_location[1] + img.shape[0], init_img.shape[0]))

            blank_image = np.zeros((img_height,img_width,3), np.uint8)

        
            if move_x > 0:
                blank_image[0:init_img.shape[0],0:init_img.shape[1]] = init_img
                blank_image[abs(cur_image_location[1]):abs(cur_image_location[1]) + img.shape[0],abs(cur_image_location[0]):abs(cur_image_location[0]) + img.shape[1]] = img
            else:
                blank_image[0:init_img.shape[0],abs(move_x):img_width] = init_img
                blank_image[abs(cur_image_location[1]):abs(cur_image_location[1]) + img.shape[0],abs(cur_image_location[0]):abs(cur_image_location[0]) + img.shape[1]] = img

        return blank_image, cur_image_location

    # Calculate the relavite position between two images.
    def calcu_cam_move(self,cur_position,pre_position):
        return [cur_position[0] - pre_position[0],cur_position[1] - pre_position[1],cur_position[2]]

    def processing_sequence(self,beg,end,cam_position,image_location):
        img_id = []
        for i in range(beg,end):
            img_id.append(i)
            if not cam_position:
                output_image = args.output + "/Blade_{}_{}.jpg".format(drone_gps_data[i][2],drone_gps_data[i][3])
                cam_position = image_stitching.get_cam_pos(drone_gps_data,i)    
                cam_rotation = image_stitching.get_cam_rot(drone_gps_data,i)
                image = image_stitching.get_img(drone_gps_data,i)

                cam_position_trans = image_stitching.world_2_camera(cam_position,cam_rotation)
                # print("init_pose:",cam_position_trans)
                if args.refinement == 'True':                    
                    detected_blade = refinment.blade_mask_detection(image,mask_rcnn_model,drone_gps_data[i][4])
                    cam_position_trans, cam_rotation = refinment.cam_refinment(image_stitching.intrinsic_matric_K,cam_position_trans, cam_rotation,image,model,detected_blade,args.visulization == 'True')
            else:
                cur_cam_position = image_stitching.get_cam_pos(drone_gps_data,i)    
                cur_cam_rotation = image_stitching.get_cam_rot(drone_gps_data,i)

                cur_cam_position_trans = image_stitching.world_2_camera(cur_cam_position,cur_cam_rotation)
                
                print("cur_cam_position_trans before refinement",cur_cam_position_trans)
                
                cur_image = image_stitching.get_img(drone_gps_data,i)
                if args.refinement == 'True':
                    detected_blade = refinment.blade_mask_detection(cur_image,mask_rcnn_model,drone_gps_data[i][4])
                    cur_cam_position_trans, cur_cam_rotation = refinment.cam_refinment(image_stitching.intrinsic_matric_K,cur_cam_position_trans,cur_cam_rotation,cur_image,model,detected_blade,args.visulization == 'True')
                    print("cur_cam_position_trans after refinement",cur_cam_position_trans)
                    cv.imwrite(os.path.join(args.output, drone_gps_data[i][4]), cur_image)

                cam_move = image_stitching.calcu_cam_move(cur_cam_position_trans,cam_position_trans)
                rot_move = np.array(cur_cam_rotation) - np.array(cam_rotation)

                dis = drone_gps_data[i][6]

                image_stitching.calculate_img_pos(cam_move,rot_move,dis , drone_gps_data[i][3] ,drone_gps_data[i][2],drone_gps_data[i][12],image_location)

                cam_position = cur_cam_position
                cam_rotation = cur_cam_rotation
                cam_position_trans = cur_cam_position_trans

        image_location = np.asarray(image_location)

        # average_x, average_y = int(np.mean(image_location, axis=0)[0]),int(np.mean(image_location, axis=0)[1])
        average_x, average_y = int(np.median(image_location, axis=0)[0]),int(np.median(image_location, axis=0)[1])
        init_img = image_stitching.get_img(drone_gps_data,img_id[0])
        pre_image_location = [0,0]
        for i in range(len(image_location)):
            move_x = image_location[i][0] if abs(image_location[i][0] - average_x) < 40 else average_x
            move_y = image_location[i][1] if abs(image_location[i][1] - average_y) < 40 else average_y
            img = image_stitching.get_img(drone_gps_data,img_id[i+1])
            init_img,pre_image_location = image_stitching.stitching(init_img,img, move_x,move_y,pre_image_location)
        cv.imwrite(output_image, init_img)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Image stitching via GPS and IMU data')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/drone/dataset/",
                        help='Directory of the GPS and IMU file')

    parser.add_argument('--output', required=True,
                        default='/home/uob/drone_localization/output',
                        metavar="/path/to/drone/output_file/",
                        help='Save the stitched image')

    parser.add_argument('--beg', required=False,
                        help="index of beginning sequence")

    parser.add_argument('--end', required=False,
                        help="index of ending sequence")

    parser.add_argument('--refinement', required=False,
                        default='False',
                        metavar="/path/to/drone/dataset/",
                        help='Use refinement algorithem to reduce data errors')

    parser.add_argument('--visulization', required=False,
                        default='False',
                        help='Visulize the point cloud projection')

    parser.add_argument('--weight', required=False,
                        default="/home/uob/drone_localization/mask_rcnn_blade_0140.h5",
                        metavar="/path/to/drone/dataset/",
                        help='Use refinement algorithem to reduce data errors')

    parser.add_argument('--point_cloud_model_path', required=False,
                        default='/home/uob/Documents/wind turbine model/manual_ritual_turbine_points.pcd',
                        metavar="/path/to/drone/point_cloud_model/",
                        help='Load the point cloud model')

    # parser.add_argument('--num_sampling', required=False,
    #                     default=14,
    #                     help="The number of data collect for one blade")

    

    args = parser.parse_args()
    
    drone_gps_data = pd.read_excel(args.dataset)
    drone_gps_data = drone_gps_data.values

    num_sampling = int(len(drone_gps_data)/12)

    image_data_path = os.path.dirname(args.dataset)

    image_stitching = Localization()


    if args.refinement == 'True': 

        model = o3d.io.read_point_cloud(args.point_cloud_model_path)
        print(args.point_cloud_model_path)
        rot = model.get_rotation_matrix_from_xyz((np.pi / 2,0,0))
        model = model.rotate(rot, center=False) #for open3d == 0.9.0
        refinment = refinement.refinement()
        mask_rcnn_model = refinment.load_mrcnn(args.weight)

    if args.beg is None:
        if args.refinement == 'True':
            for n in range(0,len(drone_gps_data),num_sampling):
                if drone_gps_data[n][3] == "LeadingEdge" or drone_gps_data[n][3] == "TrailingEdge":
                    beg, end = n + 3, n + num_sampling
                elif drone_gps_data[n][3] == "PressureSide" or drone_gps_data[n][3] == "SuctionSide":
                    beg, end = n, n + num_sampling -3

                cam_position = []
                image_location = []
                image_stitching.processing_sequence(beg,end,cam_position,image_location)
        else:
            for n in range(0,len(drone_gps_data),num_sampling):
                image_location = []
                beg, end = n, n + num_sampling
                cam_position = []
                image_stitching.processing_sequence(beg,end,cam_position,image_location)
    else:
        beg, end = int(args.beg), int(args.end)
        cam_position = []
        image_location = []
        image_stitching.processing_sequence(beg,end,cam_position,image_location)


