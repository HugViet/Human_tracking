import numpy as np

class KalmanFilter:
    # 8D state: [cx, cy, a, h, vx, vy, va, vh]
    ndim, dt = 4, 1.0

    def __init__(self):
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

        # Standard deviations
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        # measurement: [cx, cy, a, h]
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-3,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean.astype(np.float32), covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-3,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        # Project to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)  # S (4x4)

        # Innovation
        innovation = measurement - projected_mean  # (4,)

        # Compute Kalman gain K = P H^T S^{-1}
        # H is self._update_mat (4x8). So H^T is (8x4).
        PHt = covariance @ self._update_mat.T  # (8x4)

        # Use Cholesky of S for stable solve: solve(S, (PHt)^T).T
        chol = np.linalg.cholesky(projected_cov)  # (4x4)
        # First solve: L * Y = (PHt)^T -> Y shape (4x8)
        Y = np.linalg.solve(chol, PHt.T)
        # Second solve: L^T * Z = Y -> Z shape (4x8)
        Z = np.linalg.solve(chol.T, Y)
        K = Z.T.astype(np.float32)  # (8x4)

        # Update state
        new_mean = mean + K @ innovation
        new_covariance = covariance - K @ projected_cov @ K.T

        return new_mean.astype(np.float32), new_covariance.astype(np.float32)

    @staticmethod
    def gating_distance(mean, covariance, measurements, only_position=False):
        # returns squared Mahalanobis distance for each measurement
        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]
        else:
            mean = mean[:4]
            covariance = covariance[:4, :4]
            measurements = measurements[:, :4]

        cholesky = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = np.linalg.solve(cholesky, d.T)
        return np.sum(z * z, axis=0)
