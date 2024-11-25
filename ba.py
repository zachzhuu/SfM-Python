import numpy as np
import open3d as o3d
import cv2
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


# Focal length is given in my dataset
# It can be optimized as part of BA.camera_params
# See https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
# for more details
with open('./Fountain_Comp/K.txt', 'r') as f:
    K = np.array([list(map(float, line.split())) for line in f])
fx, fy = K[0, 0], K[1, 1]
focal = np.float32((fx + fy) / 2)
# My dataset is assumed to be undistorted, so I set 0 values
k = np.zeros(2, dtype=np.float32)

class Observation:
    """
    A class to represent an observation in a camera system.
    Collect 2D-3D correspondence for bundle adjustment(see ba.py).
    """
    def __init__(self, point_2d, camera_index):
        self.camera_index = camera_index
        self.point_index = None
        self.point_2d = point_2d.reshape(1, 2)

    def set_point_index(self, point_index):
        self.point_index = point_index


class BA:
    """
    Bundle Adjustment (BA) for optimizing camera parameters and 3D points.
    Attributes
    ----------
    camera_params : numpy.ndarray (n_cameras, 6)
        Initial estimates of parameters for all cameras.
        First 3 components in each row form a rotation vector 
        (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), 
        next 3 components form a translation vector, 
        then a focal distance and two distortion parameters.
    points_3d : numpy.ndarray (n_points, 3)
        Initial estimates of point coordinates in world frame.
    camera_indices : numpy.ndarray (n_observations,)
        Indices of cameras(0 to n_cameras-1) involved in each observation.
        An 'observation' means a 2D point projected on an image.
    point_indices : numpy.ndarray (n_observations,)
        Indices of 3D points(0 to n_points-1) involved in each observation.
    points_2d : numpy.ndarray (n_observations, 2)
        2D point projected on images in each observation.
    Methods
    -------
    load_data(file_name)
        Loads data from a given file and initializes the attributes.
    """
    def __init__(self):
        self.camera_params = None
        self.points_3d = None
        self.camera_indices = None
        self.point_indices = None
        self.points_2d = None

        self.n_cameras = None
        self.n_points = None

    def load_data(
        self, 
        camera_params: list[o3d.camera.PinholeCameraParameters], 
        points_3d: np.ndarray,
        obs_list: list[Observation]
    ):
        self.n_cameras = len(camera_params)
        self.n_points = points_3d.shape[0]

        cam_param_list = []
        for i in range(self.n_cameras):
            camera_param = camera_params[i]
            rotation = camera_param.extrinsic[:3, :3]                  # (3, 3)
            translation = camera_param.extrinsic[:3, 3].reshape(1, 3)  # (1, 3)
            rvec, _ = cv2.Rodrigues(rotation)
            rvec = rvec.reshape(1, 3)                                  # (1, 3)  
            cam_param_list.append(np.hstack((rvec, translation)))
        self.camera_params = np.vstack(cam_param_list)

        self.points_3d = points_3d

        self.camera_indices = np.array([obs.camera_index for obs in obs_list])
        self.point_indices = np.array([obs.point_index for obs in obs_list])
        self.points_2d = np.vstack([obs.point_2d for obs in obs_list])

        print("=========================================")
        print("Successfully loaded observations.")
        print("n_cameras: {}".format(self.n_cameras))
        print("n_points: {}".format(self.n_points))
        print("n_observations: {}".format(self.camera_indices.size))
    
    def bundle_adjustment_sparsity(self):
        n_cameras = self.n_cameras
        n_points = self.n_points
        camera_indices = self.camera_indices
        point_indices = self.point_indices

        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
        return A
    
    def run(self):
        if self.camera_params is None:
            print("No data loaded.")
            return
        
        args = (self.n_cameras, self.n_points, self.camera_indices, 
                self.point_indices, self.points_2d)
        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        f0 = fun(x0, *args)

        A = self.bundle_adjustment_sparsity()
        t0 = time.time()
        res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac',
                    ftol=1e-4, method='trf', args=args)
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

        # Plot comparison between initial and optimized residuals
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        y_min = min(f0.min(), res.fun.min())
        y_max = max(f0.max(), res.fun.max())
        axs[0].set_ylim([y_min, y_max])
        axs[1].set_ylim([y_min, y_max])
        axs[0].plot(f0, 'r', label='Initial Residuals')
        axs[0].set_title('Initial Residuals')
        axs[0].set_xlabel('Observation Index')
        axs[0].set_ylabel('Residual')
        axs[0].legend()
        axs[1].plot(res.fun, 'b', label='Optimized Residuals')
        axs[1].set_title('Optimized Residuals')
        axs[1].set_xlabel('Observation Index')
        axs[1].set_ylabel('Residual')
        axs[1].legend()
        plt.tight_layout()
        plt.show()

        opt_cam_ps = res.x[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        opt_pts = res.x[self.n_cameras * 6:].reshape((self.n_points, 3))

        return opt_cam_ps, opt_pts


def rotate(points, rot_vecs):
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) \
        + dot * (1 - cos_theta) * v
        

def rotate_points(points, rot_vecs):
    """
    Rotate points by given rotation vectors using Rodrigues' rotation formula.
    
    Parameters:
    points (np.ndarray): Array of 3D points of shape (N, 3).
    rot_vecs (np.ndarray): Array of rotation vectors of shape (N, 3).
    
    Returns:
    np.ndarray: Rotated points of shape (N, 3).
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.divide(rot_vecs, theta, out=np.zeros_like(rot_vecs), where=theta!=0)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated_points = cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    return rotated_points


def project(points, camera_params):
    """
    Convert 3-D points to 2-D by projecting onto images.
    """
    points_proj = rotate_points(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = focal
    k1 = k[0]
    k2 = k[1]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()
