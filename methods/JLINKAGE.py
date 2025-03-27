import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

def JLINKAGE_SIFT(
    filename,
    methods="JLINKAGE",
    ratio_thresh=0.6,      # Lowe或类似g2NN匹配中的比例阈值
    dist_thresh=10,        # 排除过近(自匹配)的距离阈值
    num_hypotheses=500,    # 生成的随机仿射变换模型数量（论文第3.2节：m=500）
    inlier_thresh=5.0,     # 判断匹配对是否为某模型inlier时使用的空间距离阈值
    min_cluster_pts=7,     # 过滤真正可信的变换所需的最少匹配对数 (论文中 N)
    plotimg=False,
    saveimg=False
):
    """
    复现论文中J-Linkage聚类检测拷贝粘贴的主要逻辑，输出值和HAC_SIFT等接口保持相似:
    - return: (countTrasfGeom, inliers_src_points, inliers_dst_points)

    参数:
        filename          : 输入图像文件路径
        methods           : 用于在保存可视化结果时区分文件夹名
        ratio_thresh      : 匹配时的"第二近邻比率"阈值
        dist_thresh       : 排除几乎是同一点(自匹配)的距离
        num_hypotheses    : 生成随机仿射变换模型的数量，对应论文3.2节随机采样
        inlier_thresh     : 判断某对匹配为该模型 inlier 的阈值(像素坐标距离)
        min_cluster_pts   : 论文中所说"N"值，足够多匹配点才认为是一个可信的克隆关系
        plotimg, saveimg  : 是否进行可视化展示
    """

    # ========== 第3.1节：SIFT特征提取与关键点匹配 ========== (论文中先用SIFT找关键点与描述符)
    image = cv2.imread(filename)
    if image is None:
        print(f"无法读取图像: {filename}")
        return 0, [], [], []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 OpenCV 的 SIFT 提取器
    sift = cv2.SIFT_create(
        contrastThreshold=0.01,
        edgeThreshold=50,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        print("未检测到任何SIFT特征。")
        return 0, [], [], []

    # 手动做简单的匹配(g2NN思路)，与 HAC/BIRCH 代码对应；也可改用 BFMatcher
    descriptors_norm = np.zeros_like(descriptors, dtype=np.float32)
    for i in range(len(descriptors)):
        norm_val = np.linalg.norm(descriptors[i], ord=2)
        if norm_val > 1e-8:
            descriptors_norm[i] = descriptors[i] / norm_val

    matched_kp1 = []
    matched_kp2 = []

    # 遍历每个描述符, 找它的若干近邻, 用 ratio test
    for i in range(len(descriptors_norm)):
        dotprods = descriptors_norm[i] @ descriptors_norm.T
        angles = np.arccos(np.clip(dotprods, -1.0, 1.0))  # 越小越相似
        sorted_idx = np.argsort(angles)

        j = 1
        while j < len(sorted_idx) - 1 and angles[sorted_idx[j]] < ratio_thresh * angles[sorted_idx[j+1]]:
            pt1 = np.array(keypoints[i].pt)
            pt2 = np.array(keypoints[sorted_idx[j]].pt)
            if np.linalg.norm(pt1 - pt2) > dist_thresh:
                matched_kp1.append(pt1)
                matched_kp2.append(pt2)
            j += 1

    if len(matched_kp1) == 0:
        print("没有有效的匹配对，无法进行后续聚类。")
        return 0, [], [], []

    # 令 pairs = [(x1, x2), (x1, x2), ...]
    # 其中 x1, x2都是2D坐标
    matched_pairs = list(zip(matched_kp1, matched_kp2))
    num_pairs = len(matched_pairs)

    # ========== 第3.2节：在变换参数空间上做J-Linkage聚类 ==========

    # ---- 3.2.1 随机生成若干( affine )变换假设 (论文中提到m=500) ----
    #     对每次采样：随机选取3对不共线的匹配点来估计仿射矩阵
    #     这里简化做法：随机选3对 => 用 cv2.getAffineTransform 或最小二乘DLT 算法来求解。
    #     但最好要排除三点共线等退化情况。

    # 定义一个函数: 给定3对点 => 拟合仿射矩阵(2x3)
    def estimate_affine(pts1, pts2):
        """
        利用OpenCV的estimateAffine2D做一个近似,
        或使用 findHomography + 转换, 也可。
        这里用estimateAffine2D即可。
        """
        pts1 = np.array(pts1, dtype=np.float32).reshape(-1,1,2)
        pts2 = np.array(pts2, dtype=np.float32).reshape(-1,1,2)
        M, _ = cv2.estimateAffine2D(pts1, pts2, ransacReprojThreshold=1e3, maxIters=2000)
        return M

    # 随机抽样
    rng = np.random.default_rng(seed=42)  # 固定随机种子,保证可重复
    transforms = []  # 存储估计得到的affine矩阵
    tries = 0
    max_tries = num_hypotheses * 10  # 防止极端情况死循环
    while len(transforms) < num_hypotheses and tries < max_tries:
        tries += 1
        # 随机选3对(要确保它们不共线)
        three_idx = rng.choice(num_pairs, size=3, replace=False)
        pts1 = []
        pts2 = []
        for idx in three_idx:
            x1, x2 = matched_pairs[idx]
            pts1.append(x1)
            pts2.append(x2)

        # 检测是否共线（计算面积）
        # 原理：三点共线 => 面积 ~ 0
        area = 0.5 * abs(
            pts1[0][0]*(pts1[1][1]-pts1[2][1]) +
            pts1[1][0]*(pts1[2][1]-pts1[0][1]) +
            pts1[2][0]*(pts1[0][1]-pts1[1][1])
        )
        # 若面积太小，说明可能共线 => 放弃
        if area < 1.0:
            continue

        # 尝试估计
        M = estimate_affine(pts1, pts2)
        if M is None:
            continue
        transforms.append(M)

    # ---- 3.2.2 构建匹配对的“偏好集”(Preference Set)，即每个对p是否是各变换的inlier ----
    #     论文公式(1): p是否在模型T_i内点, 若距离<阈值则PS_i(p)=1,否则=0
    # 我们将 matched_pairs 的顺序固定: 对于每个 pair p, 计算对 M_j(affine) 的内点情况
    # preference_sets 的大小: (num_pairs, len(transforms))

    preference_sets = np.zeros((num_pairs, len(transforms)), dtype=np.int32)

    def apply_affine(M, pt):
        """
        M: 2x3仿射矩阵
        pt: [x, y]
        返回 M * pt 的 [x', y']
        """
        x, y = pt
        x2 = M[0,0]*x + M[0,1]*y + M[0,2]
        y2 = M[1,0]*x + M[1,1]*y + M[1,2]
        return np.array([x2, y2], dtype=np.float32)

    # 依次对 each pair / each transform 判断
    for i, (src, dst) in enumerate(matched_pairs):
        for j, M in enumerate(transforms):
            # 用M变换src => 看与dst距离
            pred = apply_affine(M, src)
            dist = np.linalg.norm(pred - dst)
            if dist < inlier_thresh:
                preference_sets[i, j] = 1

    # ---- 3.2.2: 用 hierarchical agglomerative clustering 在 概念空间{0,1}^m 中进行聚类 ----
    #     距离使用 Jaccard 距离, 论文公式(2)。相同集合distance=0, 不相交=1
    # 我们可以先将 preference_sets 转为 bool，再对点对之间计算 jaccard 距离 => 做层次聚类

    # (1) 先转换 bool
    pref_bool = (preference_sets == 1)

    # (2) 定义一个函数计算 jaccard 距离
    def jaccard_dist(u, v):
        # u, v 是bool向量
        inter = np.count_nonzero(u & v)
        union = np.count_nonzero(u | v)
        if union == 0:
            return 1.0  # 全零向量都一样，也可认为距离为0
        return 1.0 - inter/union

    # 计算所有 pair 两两之间的 jaccard 距离
    # 注意: pdist要求我们传一个 N x dim 的 array, 这里dim=m(模型数)
    # 然而 jaccard_dist 要对 bool向量 做运算。我们可以把pref_bool.astype(int)当成int
    # 但要自定义 metric="precomputed" 可能不行 => 我们写一个自定义函数
    def jaccard_pdist(X):
        # X shape: (N, M)的bool数组
        N = X.shape[0]
        dists = np.zeros(N*(N-1)//2, dtype=np.float32)
        idx = 0
        for a in range(N-1):
            for b in range(a+1, N):
                dists[idx] = jaccard_dist(X[a], X[b])
                idx += 1
        return dists

    # 生成紧凑形式距离
    dists = jaccard_pdist(pref_bool)
    # 做层次聚类
    Z = linkage(dists, method='single')  # 论文中提到用单链, 直到Jaccard=1为分割
    clusters = fcluster(Z, t=1.0, criterion='distance')
    # clusters.shape=(num_pairs,)

    # ========== 后处理: 对每个聚类进行仿射变换校正, 统计大于 min_cluster_pts 的克隆关系 ==========
    # 这里与 HAC/BIRCH 代码的后处理类似：在同一聚类中的匹配对，我们再拟合一个变换, check inlier

    # 将全部匹配对按 cluster 分组
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for i_label, label in enumerate(clusters):
        cluster_dict[label].append(i_label)

    inliers_src_points = []
    inliers_dst_points = []
    count_geometry = 0

    # 遍历每个聚类
    for label, indices in cluster_dict.items():
        if len(indices) < min_cluster_pts:
            continue  # 太小的簇跳过

        # 收集该簇里的 (src, dst) 坐标
        cluster_src = [matched_pairs[i][0] for i in indices]
        cluster_dst = [matched_pairs[i][1] for i in indices]
        src_pts = np.array(cluster_src, dtype=np.float32)
        dst_pts = np.array(cluster_dst, dtype=np.float32)

        # 在这里再做一次 RANSAC 计算以求得更可靠的变换(类似论文3.2.2结尾)
        # 也可用 findHomography, 但仿射 => estimateAffine2D
        # threshold 适度即可
        M, mask = cv2.estimateAffine2D(src_pts.reshape(-1,1,2),
                                       dst_pts.reshape(-1,1,2),
                                       ransacReprojThreshold=4.0,
                                       maxIters=2000)
        if M is not None and mask is not None:
            inlier_count = np.sum(mask)
            if inlier_count >= min_cluster_pts:
                count_geometry += 1
                # 记录这些内点
                mask_idx = np.where(mask.ravel()>0)[0]
                inliers_src_points.extend(src_pts[mask_idx].tolist())
                inliers_dst_points.extend(dst_pts[mask_idx].tolist())

    # (count_geometry>=1) 则认为该图被检测到篡改
    if count_geometry > 0:
        print("Tampering dected!")
    else:
        print("Image not tampered.")

    # ========== 可视化输出 (可选) ==========
    if plotimg or saveimg:
        # 为了可视化，我们给每个匹配对一个颜色 = clusters[i]
        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 在图像上画线 (p1->p2)
        colors = plt.cm.get_cmap('jet', len(cluster_dict)+1)
        for i_label, label in enumerate(clusters):
            c = colors(label)  # label不同 => 不同颜色
            pt1, pt2 = matched_pairs[i_label]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=c, alpha=0.5)

        plt.title("J-Linkage clustering of SIFT matches")

        if saveimg:
            if not os.path.exists('result'):
                os.makedirs('result')
            outdir = f'result/{methods}_result'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            base_filename = os.path.basename(filename)
            outpath = os.path.join(outdir, f"detect_{base_filename}")
            plt.savefig(outpath)
            plt.close()
        else:
            plt.show()

    return count_geometry, np.array(inliers_src_points), np.array(inliers_dst_points)
