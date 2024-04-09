import numpy as np
import torch


def farthest_point_sample_numpy(point, n_point):
    """
    最远点采样基于numpy
    @param point: ndarray(2601, 3)
    @param n_point: int, 2048
    @return: index, 2048
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((n_point,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    index = centroids.astype(np.int32)
    point = point[index, :]
    return point


def farthest_point_sample_torch(args):
    """
    最远点采样
    @param args: 参数元组
    @return: 采样的点
    """
    point_data, n_point, limited_points = args
    point = point_data[:, :3].reshape(-1, 3)
    label = point_data[:, 3].reshape(-1, 1)
    device = point.device
    point_xyz = point[:, :3]
    sampled_indices = []
    if limited_points is None:
        limited_points = torch.zeros((1, 3)).to(device)
    if len(limited_points) == 0:
        limited_points = torch.zeros((1, 3)).to(device)
    while len(sampled_indices) < n_point:
        # --------------------------------------------------------------------------------------------------------------
        # 计算待采样点和限制点之间的距离
        # --------------------------------------------------------------------------------------------------------------
        distances_A = torch.sum((point_xyz[:, None, :] - limited_points[None, :, :]) ** 2, -1).sqrt()
        # --------------------------------------------------------------------------------------------------------------
        # 计算待采样点和限制点之间最近的距离
        # --------------------------------------------------------------------------------------------------------------
        min_distances_A = torch.min(distances_A, dim=1).values
        # --------------------------------------------------------------------------------------------------------------
        # 从最近的距离集合中选出最远的点，作为采样点
        # --------------------------------------------------------------------------------------------------------------
        farthest_index = torch.max(min_distances_A, -1)[1]
        # --------------------------------------------------------------------------------------------------------------
        # 将该采样点添加到采样列表中
        # --------------------------------------------------------------------------------------------------------------
        sampled_points = point_xyz[farthest_index].reshape(-1, 3)
        sampled_indices.append(farthest_index.item())
        # --------------------------------------------------------------------------------------------------------------
        # 将该采样点添加到已采样点中，以便后续计算点的距离
        # --------------------------------------------------------------------------------------------------------------
        limited_points = torch.cat((limited_points, sampled_points), dim=0)
    point_sample = point[sampled_indices]
    lable_sample = label[sampled_indices]
    point_data = torch.cat((point_sample, lable_sample), dim=1)
    return point_data
