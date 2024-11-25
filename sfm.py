import cv2
import numpy as np
import os
from glob import glob
import open3d as o3d
from ba import Observation, BA


class SfM:
    def __init__(self, image_dir, intrinsics_path):
        self.image_dir = image_dir
        self.intrinsics_path = intrinsics_path
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.matches = []
        self.good_matches = []
        self.camera_params = []
        self.all_points_3d = []
        self.all_colors = []
        self.obs_list = []
        self.K = None
        self.load_images()
        self.load_intrinsics()

    def load_images(self):
        image_paths = sorted(glob(os.path.join(self.image_dir, '*.png')), 
                             key=lambda x: int(os.path.basename(x).split('.')[0]))
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                self.images.append(img)

    def load_intrinsics(self):
        with open(self.intrinsics_path, 'r') as f:
            self.K = np.array([list(map(float, line.split())) for line in f])

    def detect_features(self):
        sift = cv2.SIFT_create()
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)

    def match_features(self):
        flann = cv2.FlannBasedMatcher()
        for i in range(len(self.descriptors) - 1):
            self.matches.append(flann.knnMatch(self.descriptors[i], self.descriptors[i + 1], k=2))

    def filter_matches(self):
        for match in self.matches:
            good = []
            for m, n in match:
                if m.distance < 0.5 * n.distance:
                    good.append(m)
            self.good_matches.append(good)

    def draw_matches(self):
        for i in range(len(self.images) - 1):
            img_matches = cv2.drawMatches(self.images[i], self.keypoints[i], self.images[i + 1], 
                                          self.keypoints[i + 1], self.good_matches[i][::10], None, 
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow(f'Matches {i}', img_matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def reconstruct(self, use_ba=False):
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=self.images[0].shape[1], 
            height=self.images[0].shape[0], 
            fx=self.K[0, 0], 
            fy=self.K[1, 1], 
            cx=self.K[0, 2], 
            cy=self.K[1, 2]
        )

        R, t = np.eye(3), np.zeros((3, 1))

        for i in range(len(self.images)):
            # save camera parameters
            extrinsics = np.hstack((R, t))
            extrinsics = np.vstack((extrinsics, np.array([0, 0, 0, 1])))
            camra_param = o3d.camera.PinholeCameraParameters()
            camra_param.extrinsic = extrinsics
            camra_param.intrinsic = camera_intrinsics
            self.camera_params.append(camra_param)

            if i == len(self.images) - 1:
                break
            
            # get 2D-2D correspondences
            pts1 = np.float32([self.keypoints[i][m.queryIdx].pt for m in self.good_matches[i]])
            pts2 = np.float32([self.keypoints[i + 1][m.trainIdx].pt for m in self.good_matches[i]])

            # estimate fundamental matrix and essential matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 
                                             ransacReprojThreshold=0.4, 
                                             confidence=0.999)
            E = self.K.T @ F @ self.K
            pts1 = pts1[mask.ravel() > 0]
            pts2 = pts2[mask.ravel() > 0]
            
            # recover relative pose from essential matrix
            _, R_new, t_new, _ = cv2.recoverPose(E, pts1, pts2, self.K)

            # first triangulation
            proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj2 = np.hstack((R_new, t_new))
            proj1 = self.K @ proj1
            proj2 = self.K @ proj2
            points_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]
            
            # transform points to world frame (first camera frame)
            points_3d = points_4d[:3] / points_4d[3]
            points_3d = R.T @ (points_3d - t)
            
            _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d.T, pts2, self.K, None, cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
            
            # second triangulation
            proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj2 = np.hstack((R_new, t_new))
            proj1 = self.K @ proj1
            proj2 = self.K @ proj2
            points_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]
            points_3d = R.T @ (points_3d - t)

            # retrieve colors for point cloud
            img = self.images[i]
            colors = []
            for pt in pts1:
                x, y = int(pt[0]), int(pt[1])
                colors.append(img[y, x])
            colors = np.array(colors)
            
            # save 3D points and colors
            self.all_points_3d.append(points_3d.T)
            self.all_colors.append(colors)
            
            # save intermediate point cloud for debugging
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d.T)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(f"./output/point_cloud_{i}.ply", pcd)
            
            # save observations for bundle adjustment
            if use_ba:
                obs1 = [Observation(pt, i) for pt in pts1]
                obs2 = [Observation(pt, i+1) for pt in pts2]
                curr_point_idx = len(self.all_points_3d)
                for j in range(len(obs1)):
                    obs1[j].set_point_index(curr_point_idx + j)
                    obs2[j].set_point_index(curr_point_idx + j)
                self.obs_list.extend(obs1)
                self.obs_list.extend(obs2)

        self.all_points_3d = np.vstack(self.all_points_3d)
        self.all_colors = np.vstack(self.all_colors)

        if use_ba:
            self.bundle_adjustment()

        self.save_results()

    def bundle_adjustment(self):
        ba = BA()
        ba.load_data(self.camera_params, self.all_points_3d, self.obs_list)
        opt_cam, opt_pts = ba.run()

        opt_cam_ps = []
        for i in range(len(self.images)):
            r_vec = opt_cam[i, :3].reshape(3, 1)
            t_vec = opt_cam[i, 3:6].reshape(3, 1)
            r_mat, _ = cv2.Rodrigues(r_vec)
            ex = np.hstack((r_mat, t_vec))
            ex = np.vstack((ex, np.array([0, 0, 0, 1])))
            opt_cam_p = o3d.camera.PinholeCameraParameters()
            opt_cam_p.extrinsic = ex
            opt_cam_p.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=self.images[0].shape[1], 
                height=self.images[0].shape[0], 
                fx=self.K[0, 0], 
                fy=self.K[1, 1], 
                cx=self.K[0, 2], 
                cy=self.K[1, 2]
            )
            opt_cam_ps.append(opt_cam_p)

        self.camera_params = opt_cam_ps
        self.all_points_3d = opt_pts

    def save_results(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.all_points_3d)
        pcd.colors = o3d.utility.Vector3dVector(self.all_colors / 255.0)

        camera_trajectory = o3d.camera.PinholeCameraTrajectory()
        camera_trajectory.parameters = self.camera_params

        o3d.io.write_point_cloud("./output/point_cloud.ply", pcd)
        o3d.io.write_pinhole_camera_trajectory("./output/camera_trajectory.json", camera_trajectory)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        for param in self.camera_params: 
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0])
            rotation = param.extrinsic[:3, :3]
            translation = param.extrinsic[:3, 3].reshape(3, 1)
            rotation = rotation.T
            translation = -rotation @ translation
            ex = np.hstack((rotation, translation))
            ex = np.vstack((ex, np.array([0, 0, 0, 1])))
            mesh_frame.transform(ex)
            vis.add_geometry(mesh_frame)
        vis.run()
        vis.destroy_window()
