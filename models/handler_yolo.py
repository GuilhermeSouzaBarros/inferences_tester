import os, date
import cv2

import fiftyone.zoo as foz
from ultralytics import YOLO

class HandlerYolo:
    def __init__(self):
        self.model_names = ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
        self.model_index = -1
        self.model_current = None

        self.sample = None
        self.sample_file = None

        self.result_output_file = None
        self.result = None

        self.threshold_iou = 0.5
        self.threshold_confidence = 0.25
        
        os.makedirs("inference_output", exist_ok=True)

        self.DIR_OUTPUT = "inference_output/yolo/" + str(date.today())
        os.makedirs(self.DIR_OUTPUT, exist_ok=True)

        self.DATASET = foz.load_zoo_dataset("coco-2017", split="validation")

    def __del__(self):
        self.output_close()

    def output_close(self):
        if self.result_output_file is not None:
            self.result_output_file.close()
            self.result_output_file = None

    def load_next_model(self) -> bool:
        self.model_index += 1
        self.output_close()

        if self.model_index < len(self.model_names):
            model_name = self.model_names[self.model_index]
            del(self.model_current)
            self.model_current = YOLO(model_name)
            print(f"\nCurrent Model: {model_name}")

            output_path = f"{self.DIR_OUTPUT}/{model_name.replace('.', '_')}.csv"
            self.result_output_file = open(output_path, "w")
            self.result_output_file.write(
                "path,num_detections,num_ground_truth," +
                "true_positives,false_positives,false_negatives," +
                "precision,recall,mAP\n")
            print(f"Output file: {output_path}")
            
            return True
        
        else:
            self.model_current = None
            return False


    def before_inference(self) -> None:
        self.sample_file = None
        while self.sample_file is None:
            self.sample = self.DATASET.take(1).first()
            if self.sample["ground_truth"] is None:
                continue
            sample_path = self.sample["filepath"]
            self.sample_file = cv2.imread(sample_path)
        print(f"\tLoaded image: {sample_path}", end="")
        
    def inference(self) -> None:
        print(f"\tDetecting objects in image...", end="")
        self.result = self.model_current.predict(
            self.sample_file,
            conf=self.threshold_confidence,
            verbose=False
        )
        

    def after_inference(self) -> None:
        detections = []
        if len(self.result) > 0 and self.result[0].boxes is not None:
            boxes = self.result[0].boxes
            for box in boxes:
                # YOLO retorna boxes normalizados (0-1)
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.DATASET.default_classes[class_id]
                
                confidence = float(box.conf[0].cpu().numpy())
                detections.append([class_name, x1, y1, x2, y2, confidence])
        
        ground_truth = self.sample["ground_truth"]
        metricas = self.calcular_metricas(detections, ground_truth)
        
        self.result_output_file.write(
            f"{self.sample['filepath']},"+
            f"{metricas['num_detections']},"+
            f"{metricas['num_ground_truth']},"+
            f"{metricas['true_positives']},"+
            f"{metricas['false_positives']},"+
            f"{metricas['false_negatives']},"+
            f"{metricas['precision']},"+
            f"{metricas['recall']},"+
            f"{metricas['mAP']}\n"
        )

        print(f"\tResult written {metricas['false_positives']}/{metricas['num_ground_truth']}", end="")





    def calcular_iou(box1, box2):
        """
        Calcula Intersection over Union (IoU) entre duas bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] ou [x_center, y_center, width, height] (normalizado)
            box2: [x1, y1, x2, y2] ou [x_center, y_center, width, height] (normalizado)
        
        Returns:
            IoU value (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcular interseção
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        area_union = area1 + area2 - area_inter
        
        if area_union == 0:
            return 0.0
        
        return area_inter / area_union

    def calcular_metricas(self, detections, ground_truth_labels):
        """
        Calcula métricas de detecção comparando com ground truth.
        
        Args:
            detections: Lista de detecções do YOLO [class_id, x1, y1, x2, y2, confidence]
            ground_truth_labels: Lista de labels YOLO [class_id, x_center, y_center, width, height]
            iou_threshold: Threshold de IoU para considerar match
        
        Returns:
            dict com métricas: num_detections, num_ground_truth, true_positives, 
                            false_positives, false_negatives, precision, recall, mAP
        """
        num_detections = len(detections)
        num_ground_truth = len(ground_truth_labels["detections"])
        
        if num_ground_truth == 0:
            # Se não há ground truth, todas as detecções são false positives
            return {
                'num_detections': num_detections,
                'num_ground_truth': 0,
                'true_positives': 0,
                'false_positives': num_detections,
                'false_negatives': 0,
                'precision': 0.0 if num_detections > 0 else 1.0,
                'recall': 0.0,
                'mAP': 0.0
            }
        
        if num_detections == 0:
            # Se não há detecções, todas as ground truth são false negatives
            return {
                'num_detections': 0,
                'num_ground_truth': num_ground_truth,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': num_ground_truth,
                'precision': 1.0,
                'recall': 0.0,
                'mAP': 0.0
            }
        
        # Converter ground truth para formato (x1, y1, x2, y2) normalizado
        gt_boxes = []
        for gt in ground_truth_labels["detections"]:
            class_name = gt["label"]
            
            x1, y1, x2, y2 = gt["bounding_box"]
            gt_boxes.append([class_name, x1, y1, x1+x2, y1+y2])
        
        # Converter detecções para formato normalizado (assumindo que já estão)
        # YOLO retorna boxes em formato (x1, y1, x2, y2) normalizado
        det_boxes = []
        for det in detections:
            class_name = det[0]
            x1, y1, x2, y2 = det[1:5]
            confidence = det[5] if len(det) > 5 else 1.0
            det_boxes.append([class_name, x1, y1, x2, y2, confidence])
        
        # Ordenar detecções por confiança (maior primeiro)
        det_boxes.sort(key=lambda x: x[5], reverse=True)
        
        # Matching: greedy matching por IoU
        matched_gt = set()
        true_positives = 0
        false_positives = 0
        
        for det in det_boxes:
            det_class, det_x1, det_y1, det_x2, det_y2, det_conf = det
            best_iou = 0.0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                
                gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                
                # Só considerar match se a classe for a mesma
                if det_class != gt_class:
                    continue
                
                iou = HandlerYolo.calcular_iou([det_x1, det_y1, det_x2, det_y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= self.threshold_iou:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives = num_ground_truth - len(matched_gt)
        
        # Calcular precision e recall
        precision = true_positives / num_detections if num_detections > 0 else 0.0
        recall = true_positives / num_ground_truth if num_ground_truth > 0 else 0.0
        
        # mAP simplificado (Average Precision) - usando precision média
        # Para mAP completo seria necessário calcular AP por classe e fazer média
        mAP = precision  # Simplificação - mAP completo requer cálculo mais complexo
        
        return {
            'num_detections': num_detections,
            'num_ground_truth': num_ground_truth,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'mAP': mAP
        }
