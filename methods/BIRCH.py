import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 新增引用
from sklearn.cluster import Birch
from itertools import combinations

def BIRCH_SIFT(
    filename,
    methods="BIRCH",
    min_cluster_pts=5,   # 最小特征点数阈值
    ratio_thresh=0.6,    # Lowe’s Ratio Test 阈值
    thc = 0.1,              # BIRCH 中的 threshold, 表示子聚类的最大半径
    bfac = 20,
    dist_thresh=10,      # 自匹配排除距离阈值
    plotimg=False, 
    saveimg=False
):
    # 读取并预处理图像
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT特征提取
    sift = cv2.SIFT_create(
        contrastThreshold=0.01,  # 对比度阈值，用于滤除低对比度区域的特征
        edgeThreshold=50,        # 边缘阈值，用于排除边缘响应较强的点
    )
    key_points, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(key_points) == 0:
        print("No SIFT keypoints found.")
        return 0, [], [], []

    # 手写最近邻匹配部分（和 HAC 中保持一致）
    descriptors_norm = np.zeros_like(descriptors, dtype=np.float32)
    for i in range(len(descriptors)):
        norm_val = np.linalg.norm(descriptors[i], ord=2)
        if norm_val > 1e-8:
            descriptors_norm[i] = descriptors[i] / norm_val

    matched_kp1, matched_kp2 = [], []
    for i in range(len(descriptors_norm)):
        dotprods = descriptors_norm[i] @ descriptors_norm.T
        angles = np.arccos(np.clip(dotprods, -1.0, 1.0))
        sorted_idx = np.argsort(angles)
        j = 1
        while j < len(sorted_idx) - 1 and angles[sorted_idx[j]] < ratio_thresh * angles[sorted_idx[j+1]]:
            pt1 = np.array(key_points[i].pt)
            pt2 = np.array(key_points[sorted_idx[j]].pt)
            if np.linalg.norm(pt1 - pt2) > dist_thresh:
                matched_kp1.append(pt1)
                matched_kp2.append(pt2)
            j += 1

    if not matched_kp1:
        print("No valid matched keypoints.")
        return 0, [], [], []

    # 准备聚类的数据
    points = np.vstack((matched_kp1, matched_kp2))  # shape (2*N, 2)

    # ========= 使用 BIRCH 算法替换层次聚类部分 ===============
    # 注意：threshold 表示子聚类直径/半径控制，可根据需求调整
    # metric 只支持 euclidean，所以在这里不使用 metric 参数
    # birch_model = Birch()
    birch_model = Birch(n_clusters=8, threshold=thc,branching_factor=bfac)
    clusters = birch_model.fit_predict(points)  # 获得每个点对应的聚类标签

    # 初始化结果
    num_gt = 0
    inliers1, inliers2 = [], []

    # 可视化匹配点与聚类（可选）
    if plotimg:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for pt1, pt2 in zip(matched_kp1, matched_kp2):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'c-', alpha=0.5)
        plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap='jet', marker='o')
        plt.title('BIRCH Clustering of SIFT Matches')
        plt.show()
    
    if saveimg:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for pt1, pt2 in zip(matched_kp1, matched_kp2):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'c-', alpha=0.5)
        plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap='jet', marker='o')
        plt.title('BIRCH Clustering of SIFT Matches')
        if not os.path.exists('result'):
            os.makedirs('result')
        if not os.path.exists(f'result\\{methods}_result'):
            os.makedirs(f'result\\{methods}_result')
        base_filename = os.path.basename(filename)
        plt.savefig(f'result\\{methods}_result\\detect_{base_filename}')
        plt.close()

    # 对每一对聚类分别计算单应性矩阵并筛选内点（与 HAC 版本保持一致）
    cluster_labels = np.unique(clusters)
    if len(cluster_labels) > 1:
        from sklearn.metrics import pairwise_distances_argmin_min  # 若需要其它处理可引入
        for c1, c2 in combinations(cluster_labels, 2):
            z1, z2 = [], []
            # 这里的逻辑：分块取 matched_kp1, matched_kp2 对应的前 N 个点和后 N 个点
            # 与 HAC 代码同理
            for idx in range(len(matched_kp1)):
                if clusters[idx] == c1 and clusters[idx + len(matched_kp1)] == c2:
                    z1.append(matched_kp1[idx])
                    z2.append(matched_kp2[idx])
                elif clusters[idx] == c2 and clusters[idx + len(matched_kp1)] == c1:
                    z1.append(matched_kp2[idx])
                    z2.append(matched_kp1[idx])

            if len(z1) > min_cluster_pts and len(z2) > min_cluster_pts:
                H, mask = cv2.findHomography(np.array(z1), np.array(z2), cv2.RANSAC, 0.05)
                if H is not None:
                    num_gt += 1
                    mask = mask.ravel().astype(bool)
                    inliers1.extend(np.array(z1)[mask].tolist())
                    inliers2.extend(np.array(z2)[mask].tolist())

    if num_gt > 0:
        print("Tampering dected!")
    else:
        print("Image not tampered.")
    
    return num_gt, np.array(inliers1), np.array(inliers2)
