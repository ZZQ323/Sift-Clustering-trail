import os
import time


def run_experiment():
    # 1. 基础配置
    methods = "BIRCH"
    db_dir = 'dataset'
    # run_F200_experiment
    DB = 'MICC-F220' 
    file_ground_truth = 'groundtruthDB_220.txt'
    
    # 使用方法
    if methods =="HAC":
        from methods.HAC import HAC_SIFT as match_features
    elif methods =="DBSCAN":
        from methods.DBSCAN import DBSCAN_SIFT as match_features
    elif methods =="BIRCH":
        from methods.BIRCH import BIRCH_SIFT as match_features
    elif methods =="JLINKAGE":
        from methods.JLINKAGE import JLINKAGE_SIFT as match_features
    else:
        from methods.HAC import HAC_SIFT as match_features
        
    # 3. 读取 ground truth
    ground_truth_path = os.path.join(db_dir, DB, file_ground_truth)
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"无法找到 ground truth 文件: {ground_truth_path}")

    # 期望文件格式: 
    # 每行内容为 "图像文件名 是否篡改(1或0)"
    # 例如: "image0001.png 1"
    with open(ground_truth_path, 'r') as f:
        lines = f.read().splitlines()

    # 储存文件名和标签信息
    image_info_list = []
    for line in lines:
        # 假设每行以空格分隔：图像名 + 0/1
        parts = line.split()
        if len(parts) != 2:
            continue
        img_name, tamper_label = parts[0], int(parts[1])
        image_info_list.append((img_name, tamper_label))

    num_images = len(image_info_list)
    print(f"总共读取到 {num_images} 个图像记录。")

    # 4. 初始化计数器
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # 记录实验用时
    tstart = time.time()

    # 5. 逐图处理
    for i, (img_name, gt_label) in enumerate(image_info_list, start=1):
        loc_file = os.path.join(db_dir, DB, img_name)
        print(f"Processing: {loc_file} ({i}/{num_images})")

        # 调用 process_image 函数 (需您自行实现对应算法)
        # 若返回值 countTrasfGeom >=1，则认为该图检测到篡改
        countTrasfGeom,*_ = match_features(loc_file,saveimg=True)
        # 与 ground truth 做比较
        detected_tampered = (countTrasfGeom >= 1)
        if detected_tampered:
            # 若检测结果为篡改
            if gt_label == 1:
                TP += 1
            else:
                FP += 1
        else:
            # 若检测结果为未篡改
            if gt_label == 1:
                FN += 1
            else:
                TN += 1

    # 6. 计算指标
    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)
    # 注意：若分母为0，需要做特殊处理以避免报错
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

    print("\nCopy-Move Forgery Detection performance:")
    print(f"TPR = {TPR*100:.2f}%")
    print(f"FPR = {FPR*100:.2f}%")

    # 7. 计算用时
    tproc = time.time() - tstart
    # 格式化输出时分秒
    hours = int(tproc // 3600)
    minutes = int((tproc % 3600) // 60)
    seconds = tproc % 60
    print(f"\nComputational time: {hours:02d}:{minutes:02d}:{seconds:04.1f}")


if __name__ == "__main__":
    run_experiment()
