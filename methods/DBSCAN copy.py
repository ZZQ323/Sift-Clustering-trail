import numpy as np
import cv2
import os
from itertools import combinations
import matplotlib.pyplot as plt

# 若使用sklearn来做DBSCAN，需要先安装 scikit-learn
from sklearn.cluster import DBSCAN

def DBSCAN_SIFT(
    filename,
    eps=40,                # DBSCAN半径
    min_samples=2,         # DBSCAN最少点数
    ratio_thresh=0.6,      # Lowe’s Ratio Test阈值
    dist_thresh=10,        # 自匹配排除的距离阈值
    ela_quality=90,        # JPEG压缩品质
    ela_thresh=25,         # 对ELA差分结果的二值化阈值
    plotimg=False,         
    saveimg=False
):
    """
    依据原论文：Copy-Move image forgery detection and classification using 
    SIFT and DBSCAN Approaches 实现的核心流程示例。

    参数:
        filename      : 输入图像路径
        eps           : DBSCAN中的eps（邻域半径）
        min_samples   : DBSCAN中的min_samples
        ratio_thresh  : Lowe’s Ratio Test的比例阈值
        dist_thresh   : 排除与自身过近的匹配，避免自匹配
        ela_quality   : ELA时JPEG重新保存的压缩质量，原论文中示例取90
        ela_thresh    : ELA差分图做二值化时的阈值，可根据实际实验略作调节
        plotimg       : 是否可视化中间匹配点与最终检测结果

    返回:
        forged_image  : 标注了可疑伪造区域和原区域的图像
        keypoints     : 提取到的SIFT关键点
        clusters      : DBSCAN得到的聚类标签（与匹配点顺序对应）
        ela_mask      : 经过ELA后最终的二值化掩码
    """

    # 1) 读取并转换为灰度图
    image = cv2.imread(filename)
    if image is None:
        raise IOError(f"无法读取图片: {filename}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) 提取SIFT特征
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(key_points) == 0:
        print("未检测到SIFT关键点。")
        return image, [], [], None

    # 3) BFMatcher进行特征匹配 + Ratio Test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # k=2 或 k>2均可，这里取 k=2 做标准的Ratio Test；也可设 k=10 再自行筛选
    matches_knn = bf.knnMatch(descriptors, descriptors, k=2)

    matched_kp1 = []
    matched_kp2 = []

    for m in matches_knn:
        # 确保取到至少两条匹配
        if len(m) < 2:
            continue
        m1, m2 = m[0], m[1]
        # Ratio Test
        if m1.distance < ratio_thresh * m2.distance:
            pt1 = key_points[m1.queryIdx].pt
            pt2 = key_points[m1.trainIdx].pt
            # 排除过近的自匹配
            if np.linalg.norm(np.array(pt1) - np.array(pt2)) > dist_thresh:
                matched_kp1.append(pt1)
                matched_kp2.append(pt2)

    if len(matched_kp1) == 0:
        print("未找到有效匹配对，可能图像无篡改或关键点不足。")
        return image, key_points, [], None

    # 4) 将匹配到的点(仅坐标)拼到一个数组中，准备做DBSCAN
    matched_kp1 = np.array(matched_kp1, dtype=np.float32)
    matched_kp2 = np.array(matched_kp2, dtype=np.float32)

    # 原论文做的是“将所有可疑点在同一空间聚类”或“分别聚类后对比”，
    # 这里示例将它们都放一起做一个整体聚类。(也可按需调试)
    all_points = np.concatenate((matched_kp1, matched_kp2), axis=0)

    # 5) DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(all_points)

    # -1 表示噪点(Outliers)
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # 不考虑噪点

    # 仅做演示：将每个簇的点画出来(可选)
    if plotimg:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_points[:, 0], all_points[:, 1],
                    c=cluster_labels, cmap='rainbow', s=10)
        plt.title("DBSCAN Clustering on Matched Keypoints")
        plt.show()
    

    # 6) 基于 ELA (Error Level Analysis) 的伪造区域高亮
    # 6.1) 先将原图保存为JPEG质量=ela_quality
    temp_jpeg = "temp_ela.jpg"
    cv2.imwrite(temp_jpeg, image, [int(cv2.IMWRITE_JPEG_QUALITY), ela_quality])
    # 再读入压缩后的图
    compressed_img = cv2.imread(temp_jpeg)
    # 删除临时文件(可选)
    if os.path.exists(temp_jpeg):
        os.remove(temp_jpeg)

    # 6.2) 做绝对差分
    ela_diff = cv2.absdiff(image, compressed_img)

    # 6.3) 转为灰度再阈值化(聚焦强度差异)
    ela_diff_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)
    _, ela_mask = cv2.threshold(ela_diff_gray, ela_thresh, 255, cv2.THRESH_BINARY)

    # 6.4) 找到差分区域的轮廓，然后进行可视化
    contours, _ = cv2.findContours(ela_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上标记
    forged_image = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 在这里可以选择性判断一下w,h是否过大或过小，以过滤噪声
        # 这里只做简单演示，直接画框
        cv2.rectangle(forged_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 这里也可结合 DBSCAN 的聚类结果，对不同簇打不同的颜色或标签
    # 本示例主要展示 ELA 产生的区域框。对原文而言，DBSCAN用于找拷贝的特征点对，
    # 而 ELA 用于进一步确定假区域 vs. 原区域。

    # 最终若需要展示处理后的图像：
    if plotimg:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(cv2.cvtColor(ela_diff, cv2.COLOR_BGR2RGB))
        axes[0].set_title("ELA 绝对差分图")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(forged_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("在原图上标记怀疑区域")
        axes[1].axis("off")
        plt.show()

    return key_points, cluster_labels, ela_mask
