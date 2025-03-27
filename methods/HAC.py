import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from itertools import combinations
import matplotlib.pyplot as plt
import os

def HAC_SIFT(
    filename,
    methods="HAC",  
    metric='euclidean',  # 
    thc=20,              # “距离小于多少”时才会把点合并到同一簇
    min_cluster_pts=5,   # 更可靠的估计，可能会希望至少 10 或更多特征点
    ratio_thresh=0.6,    # Lowe’s Ratio Test 阈值
    dist_thresh=10,      # 自匹配排除距离阈值
    plotimg=False, 
    saveimg=False
):
    # 读取并预处理图像
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT特征提取
    # sift = cv2.SIFT_create()
    sift = cv2.SIFT_create(
        contrastThreshold=0.01,  # 对比度阈值，用于滤除低对比度区域的特征
        edgeThreshold=50,       # 边缘阈值，用于排除边缘响应较强的点
    )
    key_points, descriptors = sift.detectAndCompute(gray, None)

    # 使用BFMatcher进行特征匹配
        # bf = cv2.BFMatcher(cv2.NORM_L2)
        # matches = bf.knnMatch(descriptors, descriptors, k=10)

        # # Lowe's比例测试筛选匹配点对
        # ratio_thresh = 0.6
        # matched_kp1, matched_kp2 = [], []
        # for match in matches:
        #     j = 1
        #     while j < len(match) - 1 and match[j].distance < ratio_thresh * match[j + 1].distance:
        #         temp = match[j]
        #         pt1, pt2 = key_points[temp.queryIdx].pt, key_points[temp.trainIdx].pt
        #         # 距离大于10像素防止自匹配
        #         if np.linalg.norm(np.array(pt1) - np.array(pt2)) > 10:
        #             matched_kp1.append(pt1)
        #             matched_kp2.append(pt2)
        #         j += 1

    # 手写最近邻的匹配部分
    # Step 1: 手动对特征描述符做 L2 范数归一化
    descriptors_norm = np.zeros_like(descriptors, dtype=np.float32)
    for i in range(len(descriptors)):
        norm_val = np.linalg.norm(descriptors[i], ord=2)
        if norm_val > 1e-8:  # 避免除以0
            descriptors_norm[i] = descriptors[i] / norm_val

    matched_kp1, matched_kp2 = [], []

    # 这个阈值可自行调整

    # Step 2: 使用“点积 + arccos”来计算描述符之间的角度，相当于余弦距离
    for i in range(len(descriptors_norm)):
        # dotprods: 与所有描述符做点积 => shape: (N,)
        dotprods = descriptors_norm[i] @ descriptors_norm.T
        
        # 取 arccos 之后，越小表示越相似
        angles = np.arccos(np.clip(dotprods, -1.0, 1.0))  # dot 值可能有浮动误差，需要 clip 一下
        sorted_idx = np.argsort(angles)  # 从最小角度到最大角度排序
        
        # 根据代码1里的思路，遍历排序后的结果，直到不满足 ratio test
        # (这里只是示例逻辑，可根据需要自行改动)
        j = 1
        while j < len(sorted_idx) - 1 and angles[sorted_idx[j]] < ratio_thresh * angles[sorted_idx[j+1]]:
            # 计算关键点的欧几里得距离，排除过近的自匹配
            pt1 = np.array(key_points[i].pt)
            pt2 = np.array(key_points[sorted_idx[j]].pt)
            if np.linalg.norm(pt1 - pt2) > dist_thresh:
                matched_kp1.append(pt1)
                matched_kp2.append(pt2)
            j += 1
    
    if not matched_kp1:
        return 0, [], [], []

    # 准备聚类的数据
    points = np.vstack((matched_kp1, matched_kp2))

    # 层次聚类
    Z = linkage(pdist(points), method='single', metric=metric)
    # Z = linkage(pdist(points), method='ward', metric=metric)
    clusters = fcluster(Z, t=thc, criterion='distance')

    # 初始化结果
    num_gt = 0
    inliers1, inliers2 = [], []
    cluster_labels = np.unique(clusters)

    # 可视化匹配点与聚类（可选）
    if plotimg:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for pt1, pt2 in zip(matched_kp1, matched_kp2):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'c-', alpha=0.5)
        plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap='jet', marker='o')
        plt.title('Hierarchical Clustering of SIFT Matches')
        plt.show()
    
    if saveimg:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for pt1, pt2 in zip(matched_kp1, matched_kp2):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'c-', alpha=0.5)
        plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap='jet', marker='o')
        plt.title('Hierarchical Clustering of SIFT Matches')
        if not os.path.exists('result'):
            os.makedirs('result')
        if not os.path.exists(f'result\\{methods}_result'):
            os.makedirs(f'result\\{methods}_result')
        # filename 包含完整路径
        base_filename = os.path.basename(filename)  # 例如 "example.png"
        plt.savefig(f'result\\{methods}_result\\detect_{base_filename}')  # 保存图像到 output.png 文件
        plt.close()  # 关闭图像

    # 对每一对聚类分别计算单应性矩阵并筛选内点
    if len(cluster_labels) > 1:
        for cluster1, cluster2 in combinations(cluster_labels, 2):
            z1, z2 = [], []
            for idx in range(len(matched_kp1)):
                if clusters[idx] == cluster1 and clusters[idx + len(matched_kp1)] == cluster2:
                    z1.append(matched_kp1[idx])
                    z2.append(matched_kp2[idx])
                elif clusters[idx] == cluster2 and clusters[idx + len(matched_kp1)] == cluster1:
                    z1.append(matched_kp2[idx])
                    z2.append(matched_kp1[idx])

            if len(z1) > min_cluster_pts and len(z2) > min_cluster_pts:
                H, mask = cv2.findHomography(np.array(z1), np.array(z2), cv2.RANSAC, 0.05)
                if H is not None:
                    num_gt += 1
                    mask = mask.ravel().astype(bool)
                    inliers1.extend(np.array(z1)[mask].tolist())
                    inliers2.extend(np.array(z2)[mask].tolist())
    if num_gt>0:
        print("Tampering dected!")
    else:
        print("Image not tampered.")
    return num_gt, np.array(inliers1), np.array(inliers2)

