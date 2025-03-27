import numpy as np
import cv2
import os
from itertools import combinations
import matplotlib.pyplot as plt

# 若使用sklearn来做DBSCAN，需要先安装 scikit-learn
from sklearn.cluster import DBSCAN

def DBSCAN_SIFT(
    filename,
    methods="DBSCAN",
    eps=40,              # DBSCAN 半径
    min_samples=2,       # DBSCAN 最少点数
    ratio_thresh=0.6,    # Lowe’s Ratio Test 阈值
    dist_thresh=10,      # 自匹配排除距离阈值
    ela_quality=90,      # JPEG 压缩品质 (ELA 用)
    ela_thresh=25,       # ELA 差分图二值化阈值
    plotimg=False,
    saveimg=False
):
    """
    与 HAC_SIFT 接口相似的 DBSCAN 检测算法:
      1) 提取 SIFT 特征并进行匹配
      2) 使用 DBSCAN 对所有匹配点进行聚类
      3) 通过 RANSAC 判断是否存在几何变换 (num_gt > 0 则视为检测到篡改)
      4) 使用 ELA 做误差水平分析, 并可视化可能的伪造区域 (原逻辑保留)

    返回值:
        num_gt     : 检测到的可疑几何变换数量 (≥1 表示怀疑篡改)
        inliers1   : 变换内点对应的源关键点坐标 (可能为空数组)
        inliers2   : 变换内点对应的目标关键点坐标 (可能为空数组)
    """

    # 1) 读取图像
    image = cv2.imread(filename)
    if image is None:
        print(f"无法读取图片: {filename}")
        return 0, np.array([]), np.array([])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) 提取SIFT特征
    # sift = cv2.SIFT_create()
    sift = cv2.SIFT_create(
        contrastThreshold=0.01,  # 对比度阈值，用于滤除低对比度区域的特征
        edgeThreshold=50,       # 边缘阈值，用于排除边缘响应较强的点
    )
    key_points, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(key_points) == 0:
        print("未检测到SIFT关键点。")
        return 0, np.array([]), np.array([])

    # 3) 特征匹配 + Ratio Test + 排除自匹配
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

    if len(matched_kp1) == 0:
        print("未找到足够有效的特征匹配对。")
        return 0, np.array([]), np.array([])

    # 4) 将匹配到的点合并为一个数组, 做 DBSCAN 聚类
    all_points = np.concatenate((matched_kp1, matched_kp2), axis=0)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(all_points)
    
    # 只保留非 -1 的簇
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # (可选) 绘制 DBSCAN 聚类结果
    if plotimg:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_points[:, 0], all_points[:, 1],
                    c=cluster_labels, cmap='rainbow', s=10)
        plt.title("DBSCAN clustering of matched keypoints")
        plt.show()

    if saveimg:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for pt1, pt2 in zip(matched_kp1, matched_kp2):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'c-', alpha=0.5)
        plt.scatter(all_points[:, 0], all_points[:, 1], c=cluster_labels, cmap='jet', marker='o')
        plt.title('DBSCAN of SIFT Matches')
        if not os.path.exists('result'):
            os.makedirs('result')
        if not os.path.exists(f'result\\{methods}_result'):
            os.makedirs(f'result\\{methods}_result')
        # filename 包含完整路径
        base_filename = os.path.basename(filename)  # 例如 "example.png"
        plt.savefig(f'result\\{methods}_result\\detect_{base_filename}')  # 保存图像到 output.png 文件
        plt.close()  # 关闭图像

    # 5) 继续保留原先的 ELA 误差水平分析, 只是这里不参与最终 num_gt 的计算
    # ---------------------------------------------------------------------
    temp_jpeg = "temp_ela.jpg"
    cv2.imwrite(temp_jpeg, image, [int(cv2.IMWRITE_JPEG_QUALITY), ela_quality])
    compressed_img = cv2.imread(temp_jpeg)
    if os.path.exists(temp_jpeg):
        os.remove(temp_jpeg)

    ela_diff = cv2.absdiff(image, compressed_img)
    ela_diff_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)
    _, ela_mask = cv2.threshold(ela_diff_gray, ela_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(ela_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    forged_image = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(forged_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if plotimg:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(cv2.cvtColor(ela_diff, cv2.COLOR_BGR2RGB))
        axes[0].set_title("ELA 绝对差分图")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(forged_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("在原图上标记可疑区域")
        axes[1].axis("off")
        plt.show()

    # if saveimg:
    #     if not os.path.exists('result'):
    #         os.makedirs('result')
    #     base_filename = os.path.basename(filename)
    #     save_path = os.path.join('result', f'detect_{base_filename}')
    #     cv2.imwrite(save_path, forged_image)



    # 6) 参考 HAC 的做法, 对匹配对做几何变换筛选 (RANSAC)
    #    用来统计 num_gt, 并输出内点 inliers (inliers1, inliers2)
    # ---------------------------------------------------------------------
    num_gt = 0
    inliers1, inliers2 = [], []

    n = len(matched_kp1)  # 匹配对数
    cluster_labels_arr = cluster_labels  # 长度是 2n
    # 遍历所有可能的(簇1, 簇2)组合, 并找出各自对应的匹配对
    label_list = list(unique_labels)
    if len(label_list) > 1:
        # 如果只有一个簇就没法组合, 在这里只有一个簇时也可根据需求判断是否算作可疑
        for c1, c2 in combinations(label_list, 2):
            z1, z2 = [], []
            # 遍历每对匹配
            for i in range(n):
                # 第 i 对匹配点分别位于 all_points[i], all_points[i + n]
                # 即 matched_kp1[i], matched_kp2[i]
                if (cluster_labels_arr[i] == c1 and cluster_labels_arr[i + n] == c2):
                    z1.append(matched_kp1[i])
                    z2.append(matched_kp2[i])
                elif (cluster_labels_arr[i] == c2 and cluster_labels_arr[i + n] == c1):
                    z1.append(matched_kp2[i])
                    z2.append(matched_kp1[i])

            # 根据经验, 配对数需要到达一定数量才能做变换估计, 例如 4 或更高
            if len(z1) >= 4 and len(z2) >= 4:
                H, mask = cv2.findHomography(np.array(z1), np.array(z2), cv2.RANSAC, 5.0)
                if H is not None and mask is not None:
                    mask = mask.ravel().astype(bool)
                    # 如果 RANSAC 内点足够多, 认为检测到一处可能篡改
                    num_inliers = np.sum(mask)
                    if num_inliers >= 4:
                        num_gt += 1
                        inliers1.extend(np.array(z1)[mask].tolist())
                        inliers2.extend(np.array(z2)[mask].tolist())
    else:
        # 如果只有一个非噪声簇, 看是否匹配点数足够, 也可做一次 RANSAC 试试
        c = label_list[0] if len(label_list) == 1 else None
        if c is not None:
            z1, z2 = [], []
            for i in range(n):
                if cluster_labels_arr[i] == c and cluster_labels_arr[i + n] == c:
                    z1.append(matched_kp1[i])
                    z2.append(matched_kp2[i])
            if len(z1) >= 4:
                H, mask = cv2.findHomography(np.array(z1), np.array(z2), cv2.RANSAC, 5.0)
                if H is not None and mask is not None:
                    mask = mask.ravel().astype(bool)
                    if np.sum(mask) >= 4:
                        num_gt += 1
                        inliers1.extend(np.array(z1)[mask].tolist())
                        inliers2.extend(np.array(z2)[mask].tolist())

    if num_gt > 0:
        print("Tampering detected!")
    else:
        print("Image not tampered.")

    return num_gt, np.array(inliers1), np.array(inliers2)
