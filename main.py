import cv2
import numpy as np
import os
from glob import glob
import open3d as o3d
from ba import Observation, BA


image_dir = './Foutain_Comp'
intrinsics_path = os.path.join(image_dir, 'K.txt')

def main():
    images = []
    image_paths = sorted(glob(os.path.join(image_dir, '*.png')), 
                         key=lambda x: int(os.path.basename(x).split('.')[0]))
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    with open(intrinsics_path, 'r') as f:
        K = np.array([list(map(float, line.split())) for line in f])

    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    flann = cv2.FlannBasedMatcher()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = []
    for i in range(len(descriptors) - 1):
        # matches.append(bf.match(descriptors[i], descriptors[i + 1]))
        matches.append(flann.knnMatch(descriptors[i], descriptors[i + 1], k=2))

    good_matches = []
    for match in matches:
        good = []
        for m, n in match:
            if m.distance < 0.5 * n.distance:
                good.append(m)
        good_matches.append(good)

    for i in range(len(images) - 1):
        img_matches = cv2.drawMatches(images[i], keypoints[i], images[i + 1], keypoints[i + 1], good_matches[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f'Matches {i}', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    all_points_3d = []
    all_colors = []
    camera_params = []
    obs_list = []

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=images[0].shape[1], 
        height=images[0].shape[0], 
        fx=K[0, 0], 
        fy=K[1, 1], 
        cx=K[0, 2], 
        cy=K[1, 2]
    )

    R, t = np.eye(3), np.zeros((3, 1))

    for i in range(len(images)):
        extrinsics = np.hstack((R, t))
        print(extrinsics)
        extrinsics = np.vstack((extrinsics, np.array([0, 0, 0, 1])))
        camra_param = o3d.camera.PinholeCameraParameters()
        camra_param.extrinsic = extrinsics
        camra_param.intrinsic = camera_intrinsics
        camera_params.append(camra_param)

        # Last image does not have a next image
        if i == len(images) - 1:
            break

        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches[i]])
        pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in good_matches[i]])
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        E = K.T @ F @ K
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        _, R_new, t_new, _ = cv2.recoverPose(E, pts1, pts2, K)

        proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = np.hstack((R_new, t_new))
        proj1 = K @ proj1
        proj2 = K @ proj2
        points_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)

        points_3d = points_4d[:3] / (points_4d[3])

        points_3d = R.T @ (points_3d - t)

        retval, rvec, tvec = cv2.solvePnP(points_3d.T, pts2, K, None)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        img = images[i]
        colors = []
        for pt in pts1:
            x, y = int(pt[0]), int(pt[1])
            colors.append(img[y, x])
        colors = np.array(colors)

        # for debug
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.T)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]
        o3d.io.write_point_cloud(f"point_cloud_{i}.ply", pcd)

        # Create observations for bundle adjustment
        obs1 = [Observation(pt, i) for pt in pts1]
        obs2 = [Observation(pt, i+1) for pt in pts2]
        curr_point_idx = len(all_points_3d)
        for j in range(len(obs1)):
            obs1[j].set_point_index(curr_point_idx + j)
            obs2[j].set_point_index(curr_point_idx + j)
        obs_list.extend(obs1)
        obs_list.extend(obs2)

        all_points_3d.append(points_3d.T)
        all_colors.append(colors)


    all_points_3d = np.vstack(all_points_3d)
    all_colors = np.vstack(all_colors)


    # Bundle adjustment
    ba = BA()
    ba.load_data(camera_params, all_points_3d, obs_list)
    opt_cam, opt_pts = ba.run()

    opt_cam_ps = []
    for i in range(len(images)):
        r_vec = opt_cam[i, :3].reshape(3, 1)
        t_vec = opt_cam[i, 3:6].reshape(3, 1)
        r_mat, _ = cv2.Rodrigues(r_vec)
        ex = np.hstack((r_mat, t_vec))
        print(ex)
        ex = np.vstack((ex, np.array([0, 0, 0, 1])))
        opt_cam_p = o3d.camera.PinholeCameraParameters()
        opt_cam_p.extrinsic = ex
        opt_cam_p.intrinsic = camera_intrinsics
        opt_cam_ps.append(opt_cam_p)

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(opt_pts)  # use 'all_points_3d' if no BA
    pcd.colors = o3d.utility.Vector3dVector(all_colors / 255.0)  # normalize to [0, 1]

    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    camera_trajectory.parameters = opt_cam_ps  # use 'camera_params' if no BA

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    # Save the camera trajectory to a JSON file
    o3d.io.write_pinhole_camera_trajectory("camera_trajectory.json", camera_trajectory)

    # Visualize the point cloud and camera trajectory
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for param in opt_cam_ps:  # use 'camera_params' if no BA
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


if __name__ == '__main__':
    main()
