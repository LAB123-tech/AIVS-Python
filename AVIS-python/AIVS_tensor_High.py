# -*- coding: utf-8 -*-
# @Time    : 2023-07-22
# @Author  : lab
# @desc    : 并行计算体素FPS-0：预处理，点云体素化；1：第一个组点云并行采样；2：第二个组点云并行采样；3：后处理：移除冗余点
import math
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
import open3d as o3d
from knn_cuda import KNN
from FPS import *
import torch


def delete_redundant_point(all_sampled_point, all_sampled_label, sampled_number, knn):
    delete_num = all_sampled_point.size(1) - sampled_number
    for i in range(delete_num):
        dist = knn(all_sampled_point, all_sampled_point)[0]
        min_values_one = torch.sum(dist, dim=-1)
        remain_indices = torch.argsort(min_values_one)[:, 1:]
        all_sampled_point = all_sampled_point[:, remain_indices.squeeze(), :]
        all_sampled_label = all_sampled_label[:, remain_indices.squeeze(), :]
    return torch.cat((all_sampled_point, all_sampled_label), dim=-1)


def search_limited_points(key_points, query_points, distance_threshold):
    """
    基于欧式距离的K最近邻搜索。
    @param key_points: numpy数组，形状为 (N, 3)，每一行代表一个点的坐标。
    @param query_points: numpy数组，形状为 (M, 3)，每一行代表一个查询点的坐标。
    @param distance_threshold: float，距离阈值，超过该距离的点不被考虑为最近邻。
    @return: knn_indices: numpy数组，形状为 (M, k)，表示每个查询点的K个最近邻点的索引。
    """
    limited_points = []
    for i, (query_point, key_point) in enumerate(zip(query_points, key_points)):
        distances = torch.sum((query_point[:, None, :] - key_point[None, :, :3]) ** 2, -1).sqrt()
        # --------------------------------------------------------------------------------------------------------------
        # 过滤掉距离超过阈值的点
        # --------------------------------------------------------------------------------------------------------------
        mask = distances <= distance_threshold[i]
        # --------------------------------------------------------------------------------------------------------------
        # 每一行为True的表示从所有的点中，选出体素盒子中心点对应的限制点，有的体素盒子没有限制点
        # --------------------------------------------------------------------------------------------------------------
        limited_points.append([key_point[row][:, :3] for row in mask])
    return limited_points


def visualize_point_cloud(point_cloud, point_label, file_name, voxel_box):
    """
    可视化点云
    @param point_cloud: ndarray: (1024, 3)
    @param point_label: ndarray: (1024, 1)
    @param file_name: 文件保存的名字
    @param voxel_box: 体素盒子，[AxisAlignedBoundingBox, AxisAlignedBoundingBox, ...]
    @return:
    """
    color = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
             [1.0, 0.5, 0.0], [1.0, 0.4, 1.0], [0.4, 1.0, 1.0],
             [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
             [0.5, 1.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0],
             [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.4, 0.3],
             [1.0, 0.9, 0.1], [0.5, 0.4, 0.5], [0.2, 0.1, 0.3], ]
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    # ------------------------------------------------------------------------------------------------------------------
    # 设置初始点云显示的颜色
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud_o3d.paint_uniform_color([0.5, 0.5, 0.5])
    if point_label is not None:
        # --------------------------------------------------------------------------------------------------------------
        # 根据不同标签值，将部件设定为不同的颜色
        # --------------------------------------------------------------------------------------------------------------
        min_label = np.min(point_label)
        for i in range(len(point_label)):
            point_cloud_o3d.colors[i] = color[int(point_label[i] - min_label)]
    # ------------------------------------------------------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------------------------------------------------------
    o3d.io.write_point_cloud(filename="farthest_sampled_{}.ply".format(file_name), pointcloud=point_cloud_o3d)
    if voxel_box:
        o3d.visualization.draw_geometries([point_cloud_o3d, ] + voxel_box,
                                          window_name="Open3D Window",
                                          width=1200, height=900)
    else:
        o3d.visualization.draw_geometries([point_cloud_o3d],
                                          window_name="Open3D Window",
                                          width=1200, height=900)


def visualize_point_within_voxel(point_cloud_array, point_cloud_label, voxel_boxes, file_name):
    """
    可视化体素盒子
    @param point_cloud_array: ndarray: (2601, 3)
    @param point_cloud_label: ndarray: (2601, 1)
    @param voxel_boxes: list[ndarray(17, 4), ndarray(19, 4), ...]
    @param file_name: 文件名字
    @return:
    """
    all_box = []
    for i in range(len(voxel_boxes)):
        single_voxel_box = voxel_boxes[i]
        sub_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[single_voxel_box[0, 4],
                                                                  single_voxel_box[0, 5],
                                                                  single_voxel_box[0, 6]],
                                                       max_bound=[single_voxel_box[0, 7],
                                                                  single_voxel_box[0, 8],
                                                                  single_voxel_box[0, 9]])
        sub_bbox.color = [1, 0, 0]
        all_box.extend([sub_bbox])
    # ------------------------------------------------------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------------------------------------------------------
    visualize_point_cloud(point_cloud_array, point_cloud_label, file_name, all_box)


def divide_group(voxel_boxes, sampled_rate):
    """
    将所有的体素盒子分为两组，每一个组中的体素盒子可以并行计算
    :param voxel_boxes:
    :param sampled_rate:
    :return:
    """
    batch_size = len(voxel_boxes)
    group_first, group_second = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    sample_first, sample_second = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    center_points = [[] for _ in range(batch_size)]
    flag = [set() for _ in range(batch_size)]

    for i in range(batch_size):
        voxel_list = voxel_boxes[i]
        for single_voxel in voxel_list:
            # ----------------------------------------------------------------------------------------------------------
            # 获取当前体素盒子的索引
            # ----------------------------------------------------------------------------------------------------------
            x, y, z = single_voxel[0][-3:][0].item(), single_voxel[0][-3:][1].item(), single_voxel[0][-3:][2].item()
            # ----------------------------------------------------------------------------------------------------------
            # 计算采样后的点数量
            # ----------------------------------------------------------------------------------------------------------
            voxel_sampled_after = int(len(single_voxel) * sampled_rate) + 1
            if (x, y, z) not in flag[i]:
                # ------------------------------------------------------------------------------------------------------
                # 获取当前元素的上下左右前后的索引，并去重后添加到flag中，表示这是第二组的
                # ------------------------------------------------------------------------------------------------------
                neighbors = {(x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)}
                flag[i].update(neighbors)
                group_first[i].append(single_voxel)
                sample_first[i].append(voxel_sampled_after)
            else:
                center_point = ((single_voxel[:, 7:10] + single_voxel[:, 4:7]) / 2)[0].reshape(-1, 3)
                center_points[i].append(center_point)
                group_second[i].append(single_voxel)
                sample_second[i].append(voxel_sampled_after)
    return group_first, sample_first, group_second, sample_second, center_points


def point_to_voxel(point_cloud, point_label, point_range, voxel_sizes, sample_rate):
    """
    将点放入相应的体素盒子中
    @param point_cloud: 输入的点云坐标xyz，ndarray: (2, 2048, 3)
    @param point_label: 输入的点云标签label，ndarray: (2, 2048, 1)
    @param point_range: 输入的点云坐标范围，ndarray: (2, 6)，最小值+长宽高
    @param voxel_sizes: 体素的大小
    @param sample_rate: 当前点云设置的采样率
    @return: 所有的体素盒子voxel_boxes，list[ndarray(17, 3), ndarray(19, 3), ...]，表示每个体素盒子中包含的点云
    """
    device = point_cloud.device
    # ------------------------------------------------------------------------------------------------------------------
    # 计算每个点云，长宽高各有多少个体素盒子，并且初始化空的体素盒子
    # ------------------------------------------------------------------------------------------------------------------
    xyz_number = (point_range[:, 3:] // voxel_sizes + 1).int()
    # ------------------------------------------------------------------------------------------------------------------
    # 计算每个点所属的体素盒子的索引
    # ------------------------------------------------------------------------------------------------------------------
    voxel_index = ((point_cloud - point_range[:, :3].unsqueeze(1)) // voxel_sizes.unsqueeze(1)).long()
    voxel_index_flatten = (voxel_index[:, :, 0] + voxel_index[:, :, 1] * xyz_number[:, 0].unsqueeze(1) +
                           voxel_index[:, :, 2] * xyz_number[:, 0].unsqueeze(1) * xyz_number[:, 1].unsqueeze(1))
    # ------------------------------------------------------------------------------------------------------------------
    # 获取每个点，所在体素盒子的左上角和右下角的坐标
    # ------------------------------------------------------------------------------------------------------------------
    left_top_corner = point_range[:, :3].unsqueeze(1) + voxel_index * voxel_sizes.unsqueeze(1)
    right_bottom_corner = left_top_corner + voxel_sizes.unsqueeze(1)
    # ------------------------------------------------------------------------------------------------------------------
    # 将所有数据合并到一个张量中
    # ------------------------------------------------------------------------------------------------------------------
    batch_data = torch.cat([point_cloud, point_label, left_top_corner, right_bottom_corner, voxel_index], dim=-1)
    voxel_boxes_validate_all = []
    for b in range(batch_data.shape[0]):
        single_data = batch_data[b]
        single_index = voxel_index_flatten[b]
        # --------------------------------------------------------------------------------------------------------------
        # 为每个体素盒子生成掩码，用于表示每个体素盒子包含点云中的哪些点
        # --------------------------------------------------------------------------------------------------------------
        mask = (single_index.unsqueeze(1) == torch.arange(torch.prod(xyz_number[b]).item(), device=device).unsqueeze(0))
        # --------------------------------------------------------------------------------------------------------------
        # 找到非空体素盒子的索引
        # --------------------------------------------------------------------------------------------------------------
        non_empty_indices = torch.nonzero(torch.any(mask, dim=0)).squeeze(1)
        # --------------------------------------------------------------------------------------------------------------
        # 用于存储单个点云中，非空体素盒子的点信息
        # --------------------------------------------------------------------------------------------------------------
        voxel_boxes_validate = []
        # --------------------------------------------------------------------------------------------------------------
        # 将点信息放入到非空体素盒子中
        # --------------------------------------------------------------------------------------------------------------
        for i, box_idx in enumerate(non_empty_indices):
            box_data = single_data[mask[:, box_idx]]
            voxel_boxes_validate.append(box_data)
        voxel_boxes_validate_all.append(voxel_boxes_validate)
    # ------------------------------------------------------------------------------------------------------------------
    # 将所有的体素盒子分两组，并计算每个体素盒子中，需要保留的点数量，用于并行计算
    # ------------------------------------------------------------------------------------------------------------------
    parameters = divide_group(voxel_boxes_validate_all, sample_rate)
    voxel_group_first, sample_num_first, sample_num_second, voxel_group_second, center_points = parameters
    return voxel_group_first, sample_num_first, sample_num_second, voxel_group_second, voxel_boxes_validate_all, center_points


def AVIS_Sampling(point_cloud_xyz, point_cloud_label, sampled_number):
    """
    基于AIVS结构的点云采样
    @param point_cloud_xyz: 点云数据，tensor, (2, 2048, 3)
    @param point_cloud_label: 点云标签，tensor, (1, 2048, 1)
    @param sampled_number: 采样数量，int，2048
    @return:
    """
    batch_size = point_cloud_xyz.shape[0]
    # ------------------------------------------------------------------------------------------------------------------
    # 获取采样之前的长宽高, 计算最长边为标准轴
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud_min = torch.min(point_cloud_xyz, dim=1).values
    point_cloud_max = torch.max(point_cloud_xyz, dim=1).values
    standard_axis = torch.max((point_cloud_max - point_cloud_min), dim=-1).values
    point_cloud_range = torch.cat((point_cloud_min, point_cloud_max - point_cloud_min), dim=-1)
    # ------------------------------------------------------------------------------------------------------------------
    # 设置单个体素盒子的大小，这里的除以2，不能设置太大的数字，因为math.pow(x, 1/3)可能的值很小
    # 如果这里的2设置为4，就会导致math.pow(len(point_cloud_xyz), 1 / 3)小于4，那么voxel_number就会为0
    # 但是为了防止体素盒子数量过多，可能在乘以10以后，再除以一个2，减少体素盒子的数量
    # ------------------------------------------------------------------------------------------------------------------
    voxel_number = int(math.pow((point_cloud_xyz.shape[1]), 1 / 3) / 2)
    voxel_number = voxel_number * 10 / 2
    voxel_size = ((standard_axis / voxel_number)).reshape(-1, 1)
    # ------------------------------------------------------------------------------------------------------------------
    # 预处理操作，将所有的点放入体素盒子中，并分组
    # ------------------------------------------------------------------------------------------------------------------
    start_time = time.time()
    sample_rate = sampled_number / point_cloud_xyz.shape[1]
    parameters = point_to_voxel(point_cloud_xyz, point_cloud_label, point_cloud_range, voxel_size, sample_rate)
    voxel_group_first, sampled_num_first, voxel_group_second, sampled_num_second, voxel_group_all, center_points = parameters
    end_time = time.time()
    print(f"预处理，点云体素化时间0: {end_time - start_time} 秒")
    # ------------------------------------------------------------------------------------------------------------------
    # 多线程下采样: 第一个组
    # ------------------------------------------------------------------------------------------------------------------
    start_time = time.time()
    points_end_first = []
    for idx, (voxel_group, sampled_num) in enumerate(zip(voxel_group_first, sampled_num_first)):
        args_list = list(zip_longest(voxel_group, sampled_num, []))
        with ThreadPoolExecutor(max_workers=8) as exe:
            results = exe.map(farthest_point_sample_torch, args_list)
        # --------------------------------------------------------------------------------------------------------------
        # 保存采样后的点和标签
        # --------------------------------------------------------------------------------------------------------------
        points_end_first.append([sub_list for sub_list in results])
    end_time = time.time()
    print(f"第一个组并行采样时间1: {end_time - start_time} 秒")
    points_end_first = [torch.vstack(points_end_first[i]) for i in range(batch_size)]
    # ------------------------------------------------------------------------------------------------------------------
    # 当点很稀疏的时候，体素盒子之间完全错开，就没有第二个组了
    # ------------------------------------------------------------------------------------------------------------------
    if len(voxel_group_second) == 0:
        points_end = points_end_first
    else:
        # --------------------------------------------------------------------------------------------------------------
        # 寻找每个第二组中，每个体素盒子中的限制点
        # --------------------------------------------------------------------------------------------------------------
        center_point_in_voxel_box = []
        for i in range(batch_size):
            center_point_in_voxel_box.append(torch.cat(center_points[i], dim=0))
        radius_knn = (np.sqrt(2) / 2) * voxel_size
        limited_points_current = search_limited_points(points_end_first, center_point_in_voxel_box, radius_knn)
        # --------------------------------------------------------------------------------------------------------------
        # 多线程下采样: 第二个组
        # --------------------------------------------------------------------------------------------------------------
        start_time = time.time()
        points_end_second = []
        for idx, (voxel_group, sampled_num, limited_points) in enumerate(
                zip(voxel_group_second, sampled_num_second, limited_points_current)):
            args_list = list(zip_longest(voxel_group, sampled_num, limited_points))
            with ThreadPoolExecutor(max_workers=8) as exe:
                results = exe.map(farthest_point_sample_torch, args_list)
            # --------------------------------------------------------------------------------------------------------------
            # 保存采样后的点和标签
            # --------------------------------------------------------------------------------------------------------------
            points_end_second.append([sub_list for sub_list in results])
        end_time = time.time()
        print(f"第二个组并行采样时间2: {end_time - start_time} 秒")
        points_end_second = [torch.vstack(points_end_second[i]) for i in range(batch_size)]
        # --------------------------------------------------------------------------------------------------------------
        # 将第一组和第二组的点进行组合
        # --------------------------------------------------------------------------------------------------------------
        points_end = [torch.cat((points_end_first[i], points_end_second[i])) for i in range(len(points_end_first))]
    # ------------------------------------------------------------------------------------------------------------------
    # 上述并行采样无法达到期望的采样点数量，所以增加一个后处理操作，将冗余的点去除
    # ------------------------------------------------------------------------------------------------------------------
    start_time_2 = time.time()
    knn = KNN(3, transpose_mode=True)
    for i, all_sampled_point in enumerate(points_end):
        sampled_point = all_sampled_point[:, :3].reshape(-1, 3).unsqueeze(0)
        sampled_label = all_sampled_point[:, -1].reshape(-1, 1).unsqueeze(0)
        points_end[i] = delete_redundant_point(sampled_point, sampled_label, sampled_number, knn)
    end_time_2 = time.time()
    print(f"后处理，移除冗余点4: {end_time_2 - start_time_2} 秒")
    points_end_view = points_end[0][:, :, :3].detach().cpu().numpy().squeeze()
    labels_end_view = points_end[0][:, :, -1].detach().cpu().numpy()[0]
    # ------------------------------------------------------------------------------------------------------------------
    # 可视化最后的采样结果
    # ------------------------------------------------------------------------------------------------------------------
    visualize_point_cloud(points_end_view, labels_end_view, "AIVS-{}".format(sampled_number), None)
    print("-----------------------------------------------------------------------------------------------------------")
    return points_end


def parallel_fps(point_cloud_data):
    """
    并行点云FPS
    @param point_cloud_data: 读取的点云文件，tensor, (2, 2048, 7)
    @return:
    """
    point_cloud_xyz = point_cloud_data[:, :, :3].reshape(-1, point_cloud_data.shape[1], 3)
    point_cloud_label = point_cloud_data[:, :, -1].reshape(-1, point_cloud_data.shape[1], 1)
    # visualize_point_cloud(point_cloud_xyz[0].detach().cpu().numpy(),
    #                       point_cloud_label[0].squeeze().detach().cpu().numpy(), "FPS-{}".format(512), None)
    sample_numbers = [2048, 512, 128, 32, 8]
    # ------------------------------------------------------------------------------------------------------------------
    # 使用AIVS采样
    # ------------------------------------------------------------------------------------------------------------------
    for sample_number in sample_numbers:
        points_sampled = AVIS_Sampling(point_cloud_xyz, point_cloud_label, sample_number)
        point_cloud_xyz = torch.vstack(points_sampled)[:, :, :3].reshape(-1, sample_number, 3)
        point_cloud_label = torch.vstack(points_sampled)[:, :, -1].reshape(-1, sample_number, 1)


def main():
    point_cloud_data_air = np.loadtxt(r"data/Airplane.txt").astype(np.float32)
    point_cloud_data_bag = np.loadtxt(r"data/bag.txt").astype(np.float32)
    # ------------------------------------------------------------------------------------------------------------------
    # 先进行最远点采样
    # ------------------------------------------------------------------------------------------------------------------
    point_cloud_data_air = farthest_point_sample_numpy(point_cloud_data_air, 2300)
    point_cloud_data_bag = farthest_point_sample_numpy(point_cloud_data_bag, 2300)
    point_cloud_data_air = torch.from_numpy(point_cloud_data_air).cuda().reshape(-1, len(point_cloud_data_air), 7)
    point_cloud_data_bag = torch.from_numpy(point_cloud_data_bag).cuda().reshape(-1, len(point_cloud_data_bag), 7)
    parallel_fps(torch.cat((point_cloud_data_air, point_cloud_data_bag), dim=0))
    print("Done")


if __name__ == '__main__':
    main()
