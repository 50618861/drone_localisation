"""
Visulizing drone move trajectory 


Usage: run from the command line as such:

    python3 draw_drone_trajectory.py --dataset /home/uob/Downloads/gossip-assets-files_new.xlsx --point_cloud_model_path /home/uob/Documents/wind turbine model/manual_ritual_turbine_points.pcd
"""

import open3d as o3d
import math
import numpy as np
import cv2 as cv
import pandas as pd
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='project 3D point cloud to 2D')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/drone/dataset/",
                        help='Directory of the dataset')

    parser.add_argument('--point_cloud_model_path', required=True,
                        default='/home/uob/Documents/wind turbine model/manual_ritual_turbine_points.pcd',
                        metavar="/path/to/drone/point_cloud_model/",
                        help='Load the point cloud model')

    args = parser.parse_args()

    drone_gps_data = pd.read_excel(args.dataset)
    drone_gps_data = drone_gps_data.values

    colors = [[1,0,0],[0,0,1],[1,0.7,0]]
    color_ind = 0
    position_list = []
    color_list = []


    for i in range(0,len(drone_gps_data)):

        if i% 56 == 0: # Change the color for different blade side
            color = colors[color_ind] 
            color_ind = color_ind + 1

        Eastings = drone_gps_data[i][7]
        Northings = drone_gps_data[i][8]

        x = Eastings
        y = Northings
        z = drone_gps_data[i][9]
        
        position_list.append([x,y,z])

        color_list.append(color)
    drone_position = o3d.geometry.PointCloud()
    drone_position.points = o3d.utility.Vector3dVector(position_list)
    drone_position.colors =  o3d.utility.Vector3dVector(color_list)

    model = o3d.io.read_point_cloud(args.point_cloud_model_path)
    model.paint_uniform_color([0, 0, 0])

    o3d.visualization.draw_geometries([model,drone_position])
