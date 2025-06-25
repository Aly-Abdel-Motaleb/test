import os
import subprocess

import numpy as np
import skimage.io

from datasets.base import BaseDataset
from utils.boxes import generate_anchors


class custom(BaseDataset):
    def __init__(self, phase, cfg):
        super(custom, self).__init__(phase, cfg)

        self.input_size = (768,1024)  # (height, width), both dividable by 16
        self.class_names = ('text' ,'image')
        
        self.rgb_mean = np.array([156.47688, 156.03026, 154.44823], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([72.01829 ,69.91517, 70.60924], dtype=np.float32).reshape(1, 1, 3)

        self.num_classes = len(self.class_names)
        self.class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_names)}

        self.data_dir = os.path.join(cfg.data_dir, 'custom')
        self.sample_ids, self.sample_set_path = self.get_sample_ids()

        self.grid_size = tuple(x // 16 for x in self.input_size)  # anchors grid
        
        self.anchors_seed = np.array([
                [ 87 , 34],
                [217 , 41],
                [398 , 47],
                [186 , 193],
                [614 , 64],
                [884 , 79],
                [199 , 617],
                [549 , 344],
                [892 , 541]    
            ], dtype=np.float32)
        self.anchors = generate_anchors(self.grid_size, self.input_size, self.anchors_seed)
        self.anchors_per_grid = self.anchors_seed.shape[0]
        self.num_anchors = self.anchors.shape[0]

        self.results_dir = os.path.join(cfg.save_dir, 'results')

    def get_sample_ids(self):
        sample_set_name = 'train.txt' if self.phase == 'train' \
            else 'val.txt' if self.phase == 'val' \
            else 'trainval.txt' if self.phase == 'trainval' \
            else None

        sample_ids_path = os.path.join(self.data_dir, 'image_sets', sample_set_name)
        with open(sample_ids_path, 'r') as fp:
            sample_ids = fp.readlines()
        sample_ids = tuple(x.strip() for x in sample_ids)

        return sample_ids, sample_ids_path

    def load_image(self, index):
        image_id = self.sample_ids[index]
        image_path = os.path.join(self.data_dir, 'training/image_2', image_id + '.jpg')
        image = skimage.io.imread(image_path).astype(np.float32)
        return image, image_id

    def load_annotations(self, index):
        ann_id = self.sample_ids[index]
        ann_path = os.path.join(self.data_dir, 'training/label_2', ann_id + '.txt')
        with open(ann_path, 'r') as fp:
            annotations = fp.readlines()

        annotations = [ann.strip().split(' ') for ann in annotations]
        class_ids, boxes = [], []
        for ann in annotations:
            if ann[0] not in self.class_names:
                continue
            class_ids.append(self.class_ids_dict[ann[0]])
            boxes.append([float(x) for x in ann[1:5]])

        class_ids = np.array(class_ids, dtype=np.int16)
        boxes = np.array(boxes, dtype=np.float32)

        return class_ids, boxes

    def save_results(self, results):
        txt_dir = os.path.join(self.results_dir, 'data')
        os.makedirs(txt_dir, exist_ok=True)

        for res in results:
            txt_path = os.path.join(txt_dir, res['image_meta']['image_id'] + '.txt')
            if 'class_ids' not in res:
                with open(txt_path, 'w') as fp:
                    fp.write('')
                continue

            num_boxes = len(res['class_ids'])
            with open(txt_path, 'w') as fp:
                for i in range(num_boxes):
                    class_name = self.class_names[res['class_ids'][i]]
                    score = res['scores'][i]
                    bbox = res['boxes'][i, :]
                    line = '{} {:.2f} {:.2f} {:.2f} {:.2f} {:.3f}\n'.format(
                            class_name, *bbox, score)
                    fp.write(line)

    def evaluate(self):
        """
        Evaluates detection results and calculates precision and mAP
        
        Returns:
            aps: dict containing average precision for each class and mAP
        """
        # Directory where results are saved
        txt_dir = os.path.join(self.results_dir, 'data')
        if not os.path.exists(txt_dir):
            print("No results found to evaluate")
            return {'mAP': 0.0}
            
        # For storing metrics
        aps = {}
        
        # IoU threshold for considering a detection as correct
        iou_threshold = 0.5
        
        # Process each class separately
        for class_name in self.class_names:
            # Collect all ground truth boxes for this class
            gt_boxes_all = {}
            gt_classes_all = {}
            
            # Collect all detection boxes for this class
            det_boxes_all = {}
            det_scores_all = {}
            
            # Process all samples
            for sample_id in self.sample_ids:
                # Load ground truth
                gt_path = os.path.join(self.data_dir, 'training/label_2', sample_id + '.txt')
                if os.path.exists(gt_path):
                    with open(gt_path, 'r') as f:
                        gt_lines = f.readlines()
                        
                    gt_boxes = []
                    gt_classes = []
                    
                    for line in gt_lines:
                        parts = line.strip().split(' ')
                        if parts[0] == class_name:
                            # Format: class_name xmin ymin xmax ymax
                            box = [float(x) for x in parts[1:5]]
                            gt_boxes.append(box)
                            gt_classes.append(parts[0])
                    
                    gt_boxes_all[sample_id] = np.array(gt_boxes)
                    gt_classes_all[sample_id] = gt_classes
                else:
                    gt_boxes_all[sample_id] = np.zeros((0, 4))
                    gt_classes_all[sample_id] = []
                
                # Load detections
                det_path = os.path.join(txt_dir, sample_id + '.txt')
                if os.path.exists(det_path):
                    with open(det_path, 'r') as f:
                        det_lines = f.readlines()
                        
                    det_boxes = []
                    det_scores = []
                    
                    for line in det_lines:
                        parts = line.strip().split(' ')
                        if parts[0] == class_name:
                            # Format: class_name xmin ymin xmax ymax score
                            box = [float(x) for x in parts[1:5]]
                            score = float(parts[5])
                            det_boxes.append(box)
                            det_scores.append(score)
                    
                    det_boxes_all[sample_id] = np.array(det_boxes)
                    det_scores_all[sample_id] = np.array(det_scores)
                else:
                    det_boxes_all[sample_id] = np.zeros((0, 4))
                    det_scores_all[sample_id] = np.array([])
            
            # Calculate precision for this class
            precision, ap = self._calculate_ap(gt_boxes_all, det_boxes_all, det_scores_all, iou_threshold)
            aps[class_name] = ap
            
            # Save detailed stats
            stats_dir = os.path.join(self.results_dir, 'stats')
            os.makedirs(stats_dir, exist_ok=True)
            stats_path = os.path.join(stats_dir, f'stats_{class_name.lower()}_ap.txt')
            with open(stats_path, 'w') as f:
                f.write(f'AP={ap:.6f}\n')
        
        # Calculate mAP
        aps['mAP'] = sum(aps.values()) / len(aps)
        
        return aps
    
    def _calculate_ap(self, gt_boxes_all, det_boxes_all, det_scores_all, iou_threshold=0.5):
        """
        Calculate Average Precision for a class
        
        Args:
            gt_boxes_all: dict of ground truth boxes for each sample
            det_boxes_all: dict of detection boxes for each sample
            det_scores_all: dict of detection scores for each sample
            iou_threshold: IoU threshold for considering a detection as correct
            
        Returns:
            precision: precision values at different recall points
            ap: average precision
        """
        # Flatten detections from all images and sort by confidence
        all_detections = []
        
        for sample_id in gt_boxes_all.keys():
            det_boxes = det_boxes_all[sample_id]
            det_scores = det_scores_all[sample_id]
            
            for i in range(len(det_scores)):
                all_detections.append({
                    'sample_id': sample_id,
                    'box': det_boxes[i],
                    'score': det_scores[i]
                })
        
        # Sort by decreasing confidence
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Total number of ground truth boxes
        total_gt_boxes = sum(len(boxes) for boxes in gt_boxes_all.values())
        
        if total_gt_boxes == 0:
            return [1.0], 1.0  # If no ground truth, perfect precision but meaningless
            
        if len(all_detections) == 0:
            return [0.0], 0.0  # If no detections, zero precision
        
        # Keep track of which gt boxes have been matched
        gt_matched = {sample_id: [False] * len(boxes) for sample_id, boxes in gt_boxes_all.items()}
        
        # For precision-recall curve
        true_positives = np.zeros(len(all_detections))
        false_positives = np.zeros(len(all_detections))
        
        # Process each detection in order of decreasing confidence
        for i, det in enumerate(all_detections):
            sample_id = det['sample_id']
            det_box = det['box']
            
            # Get ground truth boxes for this sample
            gt_boxes = gt_boxes_all[sample_id]
            
            if len(gt_boxes) == 0:
                false_positives[i] = 1
                continue
                
            # Calculate IoU with all ground truth boxes
            ious = self._calculate_iou(det_box, gt_boxes)
            
            # Find the best matching ground truth box
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            # If IoU exceeds threshold and not matched yet, count as true positive
            if max_iou >= iou_threshold and not gt_matched[sample_id][max_iou_idx]:
                true_positives[i] = 1
                gt_matched[sample_id][max_iou_idx] = True
            else:
                false_positives[i] = 1
        
        # Compute cumulative precision and recall
        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        
        recalls = cumulative_tp / total_gt_boxes
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
        
        # Append sentinel values for computing AP
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        
        # Ensure precision is decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Find points where recall changes
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        
        # Compute average precision (area under PR curve)
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        
        return precisions, ap
    
    def _calculate_iou(self, box, boxes):
        """
        Calculate IoU between a box and an array of boxes
        
        Args:
            box: single box [x1, y1, x2, y2]
            boxes: array of boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            ious: array of IoU values
        """
        # Box coordinates
        x1, y1, x2, y2 = box
        
        # Handle empty boxes array
        if boxes.shape[0] == 0:
            return np.array([])
        
        # Boxes coordinates
        boxes_x1 = boxes[:, 0]
        boxes_y1 = boxes[:, 1]
        boxes_x2 = boxes[:, 2]
        boxes_y2 = boxes[:, 3]
        
        # Intersection coordinates
        inter_x1 = np.maximum(x1, boxes_x1)
        inter_y1 = np.maximum(y1, boxes_y1)
        inter_x2 = np.minimum(x2, boxes_x2)
        inter_y2 = np.minimum(y2, boxes_y2)
        
        # Intersection area
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        
        # Box areas
        box_area = (x2 - x1) * (y2 - y1)
        boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1)
        
        # Union area
        union_area = box_area + boxes_area - inter_area
        
        # IoU
        iou = inter_area / union_area
        
        return iou

