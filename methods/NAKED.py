from math import sqrt
import cv2
import numpy as np
from scipy.spatial.distance import pdist

def Ransac(match_kp1, match_kp2):
    '''
    无聚类，RANSAC单次调用
    '''
    inliers1 = []
    inliers2 = []
    count, rec = 0, 0
    p1 = np.float32([kp1.pt for kp1 in match_kp1])
    p2 = np.float32([kp2.pt for kp2 in match_kp2])
    homography, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    inliers_thresold = 2.5  
    # 使用距离阈值来确定具有相等性检查的变量
    # 如果第一个关键点到第二个关键点的投影距离小于阈值，则符合单应性模型。
    # 为内点创建新的匹配数据集，以绘制所需的匹配
    good_matches = []
    for i, m in enumerate(match_kp1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        col = np.dot(homography, col)
        col /= col[2, 0]
        # 单应性与点之间的距离计算
        distance = sqrt(pow(col[0, 0] - match_kp2[i].pt[0], 2) +
                        pow(col[1, 0] - match_kp2[i].pt[1], 2))
        if distance < inliers_thresold:
            count = count + 1
    if count * 2.5 < len(match_kp1):
        inliers_thresold = 339
        rec = 3
    for i, m in enumerate(match_kp1):
            col = np.ones((3, 1), dtype=np.float64)
            col[0:2, 0] = m.pt
            col = np.dot(homography, col)
            col /= col[2, 0]
            distance = sqrt(pow(col[0, 0] - match_kp2[i].pt[0], 2) +
                        pow(col[1, 0] - match_kp2[i].pt[1], 2))
            if distance < inliers_thresold:
                good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
                inliers1.append(match_kp1[i])
                inliers2.append(match_kp2[i])
    print("匹配：", len(match_kp1))
    print("内点，即与给定单应性匹配的匹配项：", len(inliers1))
    good_points1 = np.float32([kp1.pt for kp1 in inliers1])
    good_points2 = np.float32([kp2.pt for kp2 in inliers2])
    return good_points1, good_points2, rec




def Match_features(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # origImage = image.copy() # 在绘制箭头和其它标识操作时，我们都是在原图（或其拷贝）上进行绘制，这些操作会直接修改图像数据。
    
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(gray, None)
    distance = cv2.NORM_L2
    
    bf = cv2.BFMatcher(distance)
    # 进行 K 近邻匹配，返回最近的 10 个匹配项
    matches = bf.knnMatch(descriptors, descriptors, k=10) # 一个包含多个 DMatch 对象的列表
    # Lowe 比例测试的阈值
    ratio = 0.6
    mkp1, mkp2 = [], []
    for m in matches:
        j = 1
        while m[j].distance < ratio * m[j + 1].distance:
            j = j + 1
        for k in range(1, j):
            temp = m[k]
            # 只有当距离大于 10 像素时，才认为这是一个有效的匹配，防止匹配到自己
            if pdist(np.array([key_points[temp.queryIdx].pt,
                            key_points[temp.trainIdx].pt])) > 10:
                mkp1.append(key_points[temp.queryIdx])
                mkp2.append(key_points[temp.trainIdx])
    # remove the false matches
    return Ransac(mkp1, mkp2)