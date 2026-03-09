import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

def get_segmentation_labels(data_dir):
    """
    获取 `data_dir` 文件夹中所有样本文件夹中的分割掩膜标签值。

    参数:
    - data_dir: str, 数据根目录路径，包含 BM 和 Glioma 子文件夹。
    
    返回:
    - label_set: set, 所有分割掩膜文件中的标签值集合
    """
    # 定义需要处理的类别文件夹
    categories = ["BM", "Glioma"]
    total_samples = []
    label_set = set()  # 用于存储所有标签值

    # 遍历每个类别文件夹
    for category in categories:
        category_path = os.path.join(data_dir, category)

        # 检查类别文件夹是否存在
        if not os.path.isdir(category_path):
            print(f"类别文件夹 {category_path} 不存在，跳过...")
            continue

        # 收集所有样本文件夹
        total_samples.extend(
            os.path.join(category_path, sample_folder)
            for sample_folder in os.listdir(category_path)
            if os.path.isdir(os.path.join(category_path, sample_folder))
        )

    # 用进度条处理每个样本文件夹
    for sample_path in tqdm(total_samples, desc="Collecting segmentation labels"):
        # 查找分割掩膜文件
        seg_found = False
        for file_name in os.listdir(sample_path):
            if file_name.endswith("_seg.nii.gz"):  # 分割掩膜文件的命名规则
                seg_found = True
                seg_path = os.path.join(sample_path, file_name)

                # 检查文件是否可以读取
                if not os.path.isfile(seg_path):
                    print(f"文件 {seg_path} 不存在，跳过...")
                    continue

                # 加载分割掩膜W
                image = sitk.ReadImage(seg_path)
                image_array = sitk.GetArrayFromImage(image)

                # 获取当前掩膜中的标签值
                unique_labels = np.unique(image_array)
                label_set.update(unique_labels)  # 更新标签集合

        if not seg_found:
            print(f"样本文件夹 {sample_path} 中未找到分割掩膜文件 (_seg.nii.gz)。")

    print("标签提取完成")
    return label_set



def normalize(slice, bottom=99, down=1):
    """对图像进行归一化"""
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp == tmp.min()] = -9  # 处理极小值
        return tmp



def crop_center(img, croph, cropw):
    """从图像中心裁剪出指定大小的区域"""
    height, width = img[0].shape
    starth = height//2 - (croph//2)
    startw = width//2 - (cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]



def preprocess_task1(sample_dir, sample_id):
    try:
        # 定义文件路径
        flair_path = os.path.join(sample_dir, f"{sample_id}_flair.nii.gz")
        t1ce_path = os.path.join(sample_dir, f"{sample_id}_t1ce.nii.gz")
        
        # 检查文件存在性
        if not all([os.path.exists(flair_path), os.path.exists(t1ce_path)]):
            print(f"跳过缺失文件的样本: {sample_id}")
            return False

        # 读取图像
        flair_img = sitk.ReadImage(flair_path, sitk.sitkInt16)
        t1ce_img = sitk.ReadImage(t1ce_path, sitk.sitkInt16)
        
        # 转换为数组
        flair_array = sitk.GetArrayFromImage(flair_img)  # (D,H,W)
        t1ce_array = sitk.GetArrayFromImage(t1ce_img)
        
        # 维度验证
        if flair_array.shape != t1ce_array.shape:
            print(f"维度不匹配: {sample_id} flair{flair_array.shape} vs t1ce{t1ce_array.shape}")
            return False

        # 预处理流程
        flair_norm = normalize(flair_array)
        t1ce_norm = normalize(t1ce_array)
        
        # 中心裁剪
        flair_crop = crop_center(flair_norm, 224, 224)
        t1ce_crop = crop_center(t1ce_norm, 224, 224)
        
        # 合并通道
        combined = np.stack([flair_crop, t1ce_crop], axis=-1).astype(np.float32)
        
        # 创建输出目录
        output_dir = os.path.join(sample_dir, "task1")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存npy文件
        output_path = os.path.join(output_dir, f"{sample_id}_task1.npy")
        np.save(output_path, combined)
        
        # 验证保存结果
        if not os.path.exists(output_path):
            print(f"保存失败: {sample_id}")
            return False
            
        return True
        
    except Exception as e:
        print(f"处理异常 {sample_id}: {str(e)}")
        return False



def preprocess_task2(sample_dir, sample_id, label):
    try:
        # 定义文件路径
        flair_path = os.path.join(sample_dir, f"{sample_id}_flair.nii.gz")
        t1ce_path = os.path.join(sample_dir, f"{sample_id}_t1ce.nii.gz")
        
        # 检查必须的模态文件存在性
        required_files = [flair_path, t1ce_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"缺失文件样本: {sample_id}")
            return False

        # 读取图像
        flair_img = sitk.ReadImage(flair_path, sitk.sitkInt16)
        t1ce_img = sitk.ReadImage(t1ce_path, sitk.sitkInt16)
        
        # 转换为数组并验证维度
        flair_array = sitk.GetArrayFromImage(flair_img)  # (D,H,W)
        t1ce_array = sitk.GetArrayFromImage(t1ce_img)
        
        # 处理Normal样本的分割掩膜：生成全零数组
        if label == "Normal":
            seg_array = np.zeros_like(flair_array, dtype=np.uint8)
        else:
            # 非Normal样本读取真实分割掩膜
            seg_path = os.path.join(sample_dir, f"{sample_id}_seg.nii.gz")
            if not os.path.exists(seg_path):
                print(f"分割掩膜缺失: {sample_id}")
                return False
            seg_img = sitk.ReadImage(seg_path, sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg_img)
        
        # 维度一致性检查
        if not (flair_array.shape == t1ce_array.shape == seg_array.shape):
            print(f"维度不匹配 {sample_id}: flair{flair_array.shape}, t1ce{t1ce_array.shape}, seg{seg_array.shape}")
            return False

        # 预处理流程
        flair_norm = normalize(flair_array)
        t1ce_norm = normalize(t1ce_array)
        
        # 中心裁剪
        crop_size = 224
        flair_crop = crop_center(flair_norm, crop_size, crop_size)
        t1ce_crop = crop_center(t1ce_norm, crop_size, crop_size)
        seg_crop = crop_center(seg_array, crop_size, crop_size)
        
        # 创建输出目录
        task2_dir = os.path.join(sample_dir, "task2")
        os.makedirs(task2_dir, exist_ok=True)
        
        # 清空已有文件（防止旧数据残留）
        for f in os.listdir(task2_dir):
            if f.startswith(sample_id) and f.endswith(".npy"):
                os.remove(os.path.join(task2_dir, f))
        
        for slice_idx in range(flair_crop.shape[0]):
            # 提取当前切片
            flair_slice = flair_crop[slice_idx].astype(np.float32)
            t1ce_slice = t1ce_crop[slice_idx].astype(np.float32)
            seg_slice = seg_crop[slice_idx].astype(np.uint8)
            
            
            # 合并图像模态
            image_array = np.stack([flair_slice, t1ce_slice], axis=-1)
            
            # 构建文件名
            base_name = f"{sample_id}_slice_{slice_idx:03d}"
            image_path = os.path.join(task2_dir, f"{base_name}_image.npy")
            mask_path = os.path.join(task2_dir, f"{base_name}_mask.npy")
            
            # 保存数据并验证
            np.save(image_path, image_array)
            np.save(mask_path, seg_slice)
            
            # 验证保存结果
            if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                print(f"切片保存失败: {base_name}")
                continue
                
            
        return True
        
    except Exception as e:
        print(f"处理异常 {sample_id}: {str(e)}")
        return False
    


def batch_preprocess(data_root, task_type=2, skip_normal=True):
    """统一批处理入口函数"""
    # 收集所有样本路径
    samples = []
       
    for label in ["BM", "Glioma", "Normal"]:
        label_path = os.path.join(data_root, label)
        if not os.path.exists(label_path):
            continue
            
        for sample_id in os.listdir(label_path):
            sample_dir = os.path.join(label_path, sample_id)
            if os.path.isdir(sample_dir):
                samples.append({
                    "dir": sample_dir,
                    "id": sample_id,
                    "label": label
                })

    # 处理流程
    success = 0
    skipped = 0
    failed = 0
    for sample in tqdm(samples, desc="Processing"):
        # 针对task2的特殊处理逻辑
        if task_type == 2:
            # 根据参数决定是否跳过Normal样本
            if skip_normal and sample["label"] == "Normal":
                skipped += 1
                continue
                
            # 执行task2预处理
            result = preprocess_task2(sample["dir"], sample["id"], sample["label"])  # 传递label参数
        else:
            # 其他任务处理逻辑
            result = preprocess_task1(sample["dir"], sample["id"])
            
        if result:
            success += 1
        else:
            failed += 1

    print(f"\n处理完成! 任务类型: {task_type}")
    print(f"成功样本: {success}")
    print(f"跳过Normal样本: {skipped}")
    print(f"失败样本: {failed}")
