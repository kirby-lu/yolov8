import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.yolo.utils import ops

# --- 配置参数 ---
MODEL_PATH = "weights/yolov8s.pt"
DATA_ROOT = "dataset/coco128"
IMG_SIZE = 640
CONF_THRES = 0.001  # 计算mAP通常需要极小的阈值以获取完整曲线
IOU_THRES = 0.6     # NMS 的 IoU 阈值
MAP_IOU_THRES = 0.5 # 认定为正样本的 IoU 阈值 (mAP@50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load_weights(MODEL_PATH).to(device).eval()

# --- 核心计算函数 ---

def box_iou(box1, box2):
    """计算两组边界框的 IoU"""
    def enclose_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = enclose_area(box1)
    area2 = enclose_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)

def calculate_stats(preds, targets, iou_threshold=0.5):
    """
    preds: [N, 6] -> (x1, y1, x2, y2, conf, cls)
    targets: [M, 5] -> (cls, x1, y1, x2, y2)
    """
    if len(preds) == 0:
        return np.zeros(0), np.zeros(0), targets[:, 0].tolist()

    correct = []
    detected = []
    target_cls = targets[:, 0].tolist()

    # 计算预测和真实的 IoU 矩阵
    ious = box_iou(preds[:, :4], targets[:, 1:])

    for i, p in enumerate(preds):
        p_cls = p[5]
        # 寻找类别相同且 IoU 最大的 target
        mask = (targets[:, 0] == p_cls)
        if mask.any():
            iou, idx = ious[i][mask].max(0)
            if iou > iou_threshold and idx.item() not in detected:
                correct.append(1)
                detected.append(idx.item())
                continue
        correct.append(0)
    
    return np.array(correct), preds[:, 4].cpu().detach().numpy(), target_cls

# --- 主循环 ---

def run_eval():
    img_dir = os.path.join(DATA_ROOT, "images/train2017") # coco128 默认路径
    label_dir = os.path.join(DATA_ROOT, "labels/train2017")
    
    all_stats = []
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    print(f"开始验证，共 {len(img_files)} 张图片...")
    
    for img_name in tqdm(img_files):
        # 1. 预处理 (复用你代码中的逻辑)
        img_path = os.path.join(img_dir, img_name)
        # 这里简化调用，假设你已经将 preprocess_image 放入 utils 或保持在脚本内
        # image, image_raw, h, w = preprocess_image(img_path) 
        
        # --- 快速预处理逻辑 ---
        img_raw = cv2.imread(img_path)
        h0, w0 = img_raw.shape[:2]
        r = IMG_SIZE / max(h0, w0)
        img = cv2.resize(img_raw, (int(w0 * r), int(h0 * r)))
        # Padding to 640x640 (简化版)
        img_p = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        img_p[:img.shape[0], :img.shape[1], :] = img
        img = img_p[:, :, ::-1].transpose(2, 0, 1) # BGR2RGB, HWC2CHW
        img = np.ascontiguousarray(img)
        input_ = torch.from_numpy(img).to(device).float() / 255.0
        input_ = input_.unsqueeze(0)

        # 2. 推理
        with torch.no_grad():
            preds = model(input_)
        
        # 3. 后处理 (NMS)
        preds = ops.non_max_suppression(preds, CONF_THRES, IOU_THRES)[0]
        
        # 4. 读取真实标签 (Label)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            continue
            
        # 加载标注并还原坐标 (YOLO格式: cls, x_c, y_c, w, h -> x1, y1, x2, y2)
        with open(label_path, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        
        if len(labels) > 0:
            # 还原到 640 缩放后的坐标系计算 IoU
            t_boxes = labels[:, 1:].copy()
            # 简单还原逻辑：根据预处理的缩放比例 r
            t_boxes[:, [0, 2]] *= w0 * r 
            t_boxes[:, [1, 3]] *= h0 * r
            # xywh2xyxy
            new_boxes = np.zeros_like(t_boxes)
            new_boxes[:, 0] = t_boxes[:, 0] - t_boxes[:, 2] / 2
            new_boxes[:, 1] = t_boxes[:, 1] - t_boxes[:, 3] / 2
            new_boxes[:, 2] = t_boxes[:, 0] + t_boxes[:, 2] / 2
            new_boxes[:, 3] = t_boxes[:, 1] + t_boxes[:, 3] / 2
            
            targets = torch.cat([torch.tensor(labels[:, :1]), torch.tensor(new_boxes)], dim=1).to(device)
            
            # 5. 计算单图统计数据
            stats = calculate_stats(preds, targets, MAP_IOU_THRES)
            all_stats.append(stats)

    # 6. 计算最终 mAP
    correct = np.concatenate([x[0] for x in all_stats])
    conf = np.concatenate([x[1] for x in all_stats])
    tcls = np.concatenate([x[2] for x in all_stats])
    
    # 按照置信度排序计算 PR 曲线
    i = np.argsort(-conf)
    correct, conf = correct[i], conf[i]
    
    # 计算召回率和精度
    tp = np.cumsum(correct)
    fp = np.cumsum(1 - correct)
    
    recall = tp / (len(tcls) + 1e-16)
    precision = tp / (tp + fp)
    
    # 计算 AP (Area Under Curve)
    ap = 0
    for t in np.arange(0, 1.1, 0.1): # 11点插值法 (COCO旧标准) 或全积分
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    print(f"\n手动验证结果:")
    print(f"mAP@50: {ap:.4f}")

if __name__ == "__main__":
    run_eval()