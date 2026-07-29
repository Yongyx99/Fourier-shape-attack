"""
tools in test functions
"""
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculating IoU of box1 and box2
    box1: [x1, y1, x2, y2]
    box2: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(detections, ground_truths, iou_threshold=0.7):
    """
    Calculating Average Precision (AP)
    Args:
        detections: detection results list, ['boxes','scores','image_id']
        ground_truths: GT list, ['boxes','image_id']
        iou_threshold: IoU threshold
    Returns:
        ap: AP value
        recalls (np.array): recall values used for ploting
        precisions (np.array): precision values used for ploting
    """
    gt_dict = {}
    for gt in ground_truths:
        img_id = gt['image_id']
        gt_dict[img_id] = {'boxes': gt['boxes'], 'matched': [False] * len(gt['boxes'])}
    
    # Collect and sort all test boxes (in descending order of confidence)
    all_detections = []
    for det in detections:
        img_id = det['image_id']
        for box, score in zip(det['boxes'], det['scores']):
            all_detections.append({
                'image_id': img_id,
                'box': box,
                'score': score
            })
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    
    #  initializing tp and fp
    tp_list = np.zeros(len(all_detections))
    fp_list = np.zeros(len(all_detections))
    for i, det in enumerate(all_detections):
        img_id = det['image_id']
        box = det['box']
        
        # If the current image has no GT, it is marked as FP
        if img_id not in gt_dict:
            fp_list[i] = 1
            continue
        gt_info = gt_dict[img_id]
        best_iou = 0.0
        best_gt_idx = -1
        
        # Calculating IoU of GT boxes
        for gt_idx, gt_box in enumerate(gt_info['boxes']):
            if gt_info['matched'][gt_idx]:
                continue
            iou = calculate_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp_list[i] = 1
            gt_info['matched'][best_gt_idx] = True  # 标记GT为已匹配
        else:
            fp_list[i] = 1
    
    cum_tp = np.cumsum(tp_list)
    cum_fp = np.cumsum(fp_list)
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)
    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / total_gt

    recalls_expanded = np.concatenate(([0], recalls))
    precisions_expanded = np.concatenate(([1], precisions))
       
    # calculating AP
    ap = 0
    for i in range(1, len(recalls_expanded)):
        delta_r = recalls_expanded[i] - recalls_expanded[i-1]
        ap += delta_r * precisions_expanded[i]
    
    return ap, recalls_expanded, precisions_expanded
def plot_PRcurve(orig_recalls, orig_precisions, orig_ap, work_dir):
    """
    plot PR curve
    Args:
        orig_recalls (np.array): recall values of original detections
        orig_precisions (np.array): precision values of original detections
        orig_ap (float): original AP value
        work_dir (str): path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(orig_recalls, orig_precisions, 'b-', linewidth=2, label=f'Original (AP={orig_ap:.4f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Original Detection Precision-Recall Curve', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'original_pr_curve.png'), dpi=300)
    plt.show()
def plot_comparison_pr(orig_recalls, orig_precisions, orig_ap, 
                       adv_recalls, adv_precisions, adv_ap, 
                       work_dir):
    """绘制原始与对抗结果的PR曲线对比"""
    plt.figure(figsize=(10, 8))
    plt.plot(orig_recalls, orig_precisions, 'b-', linewidth=2, label=f'Original (AP={orig_ap:.4f})')
    plt.plot(adv_recalls, adv_precisions, 'r--', linewidth=2, label=f'Adversarial (AP={adv_ap:.4f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves Comparison', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    # 添加AP差值信息
    ap_drop = orig_ap - adv_ap
    plt.text(0.7, 0.05, f'AP Drop: {ap_drop:.4f}', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'pr_curve_comparison.png'), dpi=300)
    plt.show()