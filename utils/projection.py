"""
Projecting 3D point cloud model to 2D image plane


Usage: run from the command line as such:

    python3 projection.py --dataset /home/uob/Downloads/gossip-assets-files_new.xlsx --output /home/uob/drone_localization/output --point_cloud_model_path /home/uob/Documents/wind turbine model/manual_ritual_turbine_points.pcd
"""

import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2 as cv
import open3d as o3d
import pandas as pd
import os

def calculate_point(intrinsic_matric_K,cam_pose,model_point):
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        projected_cuboid = intrinsic_matric_K.dot(T)
        projected_cuboid = projected_cuboid.dot(cam_pose)
        projected_cuboid = projected_cuboid.dot(model_point.T)

        result = projected_cuboid.T

        result[...,0] = result[...,0]/result[...,2]
        result[...,1] = result[...,1]/result[...,2]
        result[...,2] = result[...,2]/result[...,2]

        new_point = result[...,:2]

        return  np.asarray(new_point)

def pose_transform(trans, rot):
    r = R.from_euler("xzy", rot, degrees=True)
    rotation_matrix = r.as_matrix()
    rotation_matrix = np.linalg.inv(rotation_matrix)
    # print(rotation_matrix,trans)
    pose = np.r_['0,2',np.c_[rotation_matrix,trans],np.asarray([0,0,0,1])]

    return pose


def visulize_proj(intrinsic_matric_K,trans,rot,img,model,color):
    model_point = np.asarray(model.points)    
    model_point = np.c_[model_point, np.ones(len(model_point))]
    cam_pose = pose_transform(trans,rot)
    blade_points = calculate_point(intrinsic_matric_K,cam_pose,model_point)
    for point in blade_points:
        if 0 < point[0] < img.shape[1] and 0 < point[1] < img.shape[0]:
            cv.circle(img, (int(point[0]),int(point[1])),1 ,color, -1)
    return img

def resize (img, width, height):
    dim = (width, height)    
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    return img

def get_img(data,num):
    img = cv.imread("/home/uob/Documents/dataset/gossip-assets-files-localisation/" + data[num][4])
    img = resize(img, 640, 480)
    return img

def get_cam_rot(data,num):   
    return [data[num][10],data[num][11],data[num][12]]

def get_cam_pos(data,num):
    return [data[num][8],data[num][7],data[num][9]]

def world_2_camera(drone_tran,drone_rot):
    t = np.array([drone_tran[1],-drone_tran[2],drone_tran[0]])
    r = R.from_euler("xzy", drone_rot, degrees=True)

    rotation_matrix = r.as_matrix()
    rotation_matrix = np.linalg.inv(rotation_matrix)

    trans_t = rotation_matrix.dot(t.T)
    return -trans_t

if __name__ == '__main__':

     # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='project 3D point cloud to 2D')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/drone/dataset/",
                        help='Directory of the dataset')

    parser.add_argument('--output', required=True,
                        default='/home/uob/drone_localization/output',
                        metavar="/path/to/drone/output_file/",
                        help='Save the stitched image')

    parser.add_argument('--point_cloud_model_path', required=True,
                        default='/home/uob/Documents/wind turbine model/manual_ritual_turbine_points.pcd',
                        metavar="/path/to/drone/point_cloud_model/",
                        help='Load the point cloud model')

    intrinsic_matric_K = np.array([
                                    [1787.69, 0, 320],
                                    [0, 2008.34, 240],
                                    [0, 0, 1]
                                ], dtype='f') 


    args = parser.parse_args()
    drone_gps_data = pd.read_excel(args.dataset)
    drone_gps_data = drone_gps_data.values


    model = o3d.io.read_point_cloud(args.point_cloud_model_path)
    rot = model.get_rotation_matrix_from_xyz((np.pi / 2,0,0))
    model = model.rotate(rot, center=(0, 0, 0))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i in range(len(drone_gps_data)):

        cam_position = get_cam_pos(drone_gps_data,i)
        cam_rotation = get_cam_rot(drone_gps_data,i)
        image = get_img(drone_gps_data,i)
        cam_position_trans = world_2_camera(cam_position,cam_rotation)

        img = visulize_proj(intrinsic_matric_K,cam_position_trans,cam_rotation,image,model,color = [0,255,0])

        cv.imwrite(os.path.join(args.output,drone_gps_data[i][4]), img)



        