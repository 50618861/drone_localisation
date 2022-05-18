"""
Refinement Algrithm

"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import os,sys

import tensorflow as tf

# Root directory of mask_rcnn
ROOT_DIR = os.path.abspath("/home/uob/Documents/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import skimage

from blade import blade_detection

class refinement():

    Max_iteration = 10

    def calculate_point(self, intrinsic_matric_K,cam_pose,model_point):
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

    def pose_transform(self, trans, rot):
        r = R.from_euler("xzy", rot, degrees=True)
        rotation_matrix = r.as_matrix()
        rotation_matrix = np.linalg.inv(rotation_matrix)
        # print(rotation_matrix,trans)
        pose = np.r_['0,2',np.c_[rotation_matrix,trans],np.asarray([0,0,0,1])]

        return pose

    def fproject(self,intrinsic_matric_K,trans,rot,img,model):
        model_point = np.asarray(model.points)    
        model_point = np.c_[model_point, np.ones(len(model_point))]
        cam_pose = self.pose_transform(trans,rot)
        blade_points = self.calculate_point(intrinsic_matric_K,cam_pose,model_point)
        output = []
        for point in blade_points:
            if 0 < point[0] < img.shape[1] and 0 < point[1] < img.shape[0]:
                output.append(point)
        return np.asarray(output)

    def visulize_proj(self,intrinsic_matric_K,trans,rot,img,model,color):
        model_point = np.asarray(model.points)    
        model_point = np.c_[model_point, np.ones(len(model_point))]
        cam_pose = self.pose_transform(trans,rot)
        blade_points = self.calculate_point(intrinsic_matric_K,cam_pose,model_point)
        output = []
        for point in blade_points:
            if 0 < point[0] < img.shape[1] and 0 < point[1] < img.shape[0]:
                output.append(point)
                cv.circle(img, (int(point[0]),int(point[1])),1 ,color, -1)
        return np.asarray(output)

    def load_mrcnn(self, weights_path = "/home/uob/blade_detect/mask_rcnn_blade_0140.h5"):
        # Directory to save logs and trained model
        LOGS_DIR = os.path.join(ROOT_DIR, "logs")

        # Inference Configuration
        config = blade_detection.BladeInferenceConfig()
        config.display()

        DEVICE = "gpu:0"  # /cpu:0 or /gpu:0
        TEST_MODE = "inference"

        # Create model in inference mode
        with tf.device(DEVICE):
            mask_rcnn_model = modellib.MaskRCNN(mode="inference",
                                    model_dir=LOGS_DIR,
                                    config=config)
        # Load weights
        print("Loading weights ", weights_path)
        mask_rcnn_model.load_weights(weights_path, by_name=True)

        return mask_rcnn_model

    def blade_mask_detection(self, img, mask_rcnn_model, visulize_mask=False):

        r = mask_rcnn_model.detect([img], verbose=0)[0]
        mask = r['masks']
        points = np.where(mask==True)

        if visulize_mask:
            visualize.display_instances(
                img, r['rois'], r['masks'], r['class_ids'],
                "test", r['scores'],
                show_bbox=True, show_mask=True,
                title="Predictions")

        return points

    def point_pair(self,nbrs,points,reference_points):

        _, indices = nbrs.kneighbors(points)
        refer = np.squeeze(reference_points[indices])

        return refer

    def cam_refinment(self,intrinsic_matric_K,drone_tran,drone_rot,img,model,points,visualization):
        if visualization:
            num_point = self.visulize_proj(intrinsic_matric_K,drone_tran,drone_rot,img,model,[0,255,255])
        else:
            num_point = self.fproject(intrinsic_matric_K,drone_tran,drone_rot,img,model)

        if points[0].shape[0] == 0:
            print("Blade segmentation failed")
            return drone_tran, drone_rot

        random_select = np.random.choice(points[0].shape[0],num_point.shape[0],replace=False)
        point_list = []

        for i in random_select:
            point_list.append([points[1][i],points[0][i]])
            if visualization:
                cv.circle(img, [points[1][i],points[0][i]],1 ,[0,0,255], -1)

        point_list = np.asarray(point_list)

        nbrs = NearestNeighbors(n_neighbors=1, radius = 10,algorithm='auto').fit(point_list)
        J = []

        e = 0.00000001

        # Only refine the postion
        for i in range(self.Max_iteration):
            blade_points = self.fproject(intrinsic_matric_K,drone_tran,drone_rot,img,model)
            refer = self. point_pair(nbrs,blade_points,point_list)

            J_1 = np.array(((self.fproject(intrinsic_matric_K,drone_tran + [e,0,0],drone_rot,img,model)).flatten() - blade_points.flatten())/e)
            J_2 = np.array(((self.fproject(intrinsic_matric_K,drone_tran + [0,e,0],drone_rot,img,model)).flatten() - blade_points.flatten())/e)
            J_3 = np.array(((self.fproject(intrinsic_matric_K,drone_tran + [0,0,e],drone_rot,img,model)).flatten() - blade_points.flatten())/e)

            J = np.asarray([J_1, J_2, J_3])

            dy = refer.flatten() - blade_points.flatten()
            dx = np.linalg.pinv(J.T).dot(dy.T)
            norm   = np.linalg.norm(dy, axis=0)

            drone_tran = drone_tran + dx[0:3]


        # refine the postion and rotation
        # for i in range(10):
        #     blade_points = self.fproject(intrinsic_matric_K,drone_tran,drone_rot,img,model)
        #     # print(blade_points)
        #     refer = self.point_pair(nbrs,blade_points,point_list)

        #     J_1 = np.array(((self.fproject(intrinsic_matric_K,drone_tran + [e,0,0],drone_rot,img,model)).flatten() - blade_points.flatten())/e)
        #     J_2 = np.array(((self.fproject(intrinsic_matric_K,drone_tran + [0,e,0],drone_rot,img,model)).flatten() - blade_points.flatten())/e)
        #     J_3 = np.array(((self.fproject(intrinsic_matric_K,drone_tran + [0,0,e],drone_rot,img,model)).flatten() - blade_points.flatten())/e)

        #     J_4 = np.array(((self.fproject(intrinsic_matric_K,drone_tran, drone_rot + [e,0,0],img,model)).flatten() - blade_points.flatten())/e)
        #     J_5 = np.array(((self.fproject(intrinsic_matric_K,drone_tran, drone_rot + [0,e,0],img,model)).flatten() - blade_points.flatten())/e)
        #     # J_6 = np.array(((fproject(intrinsic_matric_K,drone_tran, drone_rot + [0,0,e],img,model)).flatten() - blade_points.flatten())/e)

        #     # J = np.asarray([J_1, J_2, J_3,J_4, J_5, J_6])

        #     J = np.asarray([J_1, J_2, J_3,J_4, J_5])

        #     dy = refer.flatten() - blade_points.flatten()
        #     dx = np.linalg.pinv(J.T).dot(dy.T)
        #     # print(dx)
        #     norm   = np.linalg.norm(dy, axis=0)

        #     drone_tran = drone_tran + dx[0:3]
        #     drone_rot[0:2] = drone_rot[0:2] + dx[3:5]
        #     # drone_rot = drone_rot + dx[3:6]
        #     # print(drone_tran,drone_rot)
        #     if abs(np.linalg.norm(dx[0:3])/np.linalg.norm(drone_tran) + np.linalg.norm(dx[3:6])/np.linalg.norm(drone_rot)) <= 0.0005:
        #         break

        if visualization:
            num_point = self.visulize_proj(intrinsic_matric_K,drone_tran,drone_rot,img,model,color = [255,255,0])

        return drone_tran, drone_rot
