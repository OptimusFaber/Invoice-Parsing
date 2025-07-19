import layoutparser as lp
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from typing import Union, List, Dict

class Parser:
    def __init__(
        self,
        layout_config_path: str,
        layout_model_path: str,
        yolo_model_path: str,
        label_map: Dict[int, str],
        color_map: Dict[str, str],
        layout_confidence: float = 0.5,
        layout_nms: float = 0.7,
        yolo_confidence: float = 0.5,
        yolo_nms: float = 0.7,
        cuda: bool = False,
        noise_level: int = 0
    ):
        """
        Инициализация парсера документов.
        
        Args:
            layout_config_path: путь к конфигурации LayoutParser
            layout_model_path: путь к весам LayoutParser
            yolo_model_path: путь к весам YOLO
            label_map: словарь соответствия индексов классов их названиям
            color_map: словарь соответствия названий классов их цветам
            layout_confidence: порог уверенности для LayoutParser
            layout_nms: порог NMS для LayoutParser
            yolo_confidence: порог уверенности для YOLO
            yolo_nms: порог NMS для YOLO
            cuda: использовать ли CUDA
            noise_level: уровень вывода информации (0 - тихий режим, 1 - подробный)
        """
        self.device = 'cuda:0' if cuda and torch.cuda.is_available() else 'cpu'
        self.noise_level = noise_level
        
        if self.noise_level > 0:
            print(f"Используется устройство: {self.device}")
            
            if self.device.startswith('cuda'):
                print(f"Активное CUDA устройство: {torch.cuda.get_device_name(0)}")
                print(f"Доступная память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Инициализация LayoutParser
        self.layout_model = lp.models.Detectron2LayoutModel(
            config_path=layout_config_path,
            model_path=layout_model_path,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", layout_confidence,
                "MODEL.ROI_HEADS.NMS_THRESH_TEST", layout_nms,
                "MODEL.DEVICE", self.device
            ],
            label_map=label_map
        )
        
        # Инициализация YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.conf = yolo_confidence
        self.yolo_model.iou = yolo_nms
        self.yolo_model.to(self.device)
        
        # Сохраняем цветовую карту
        self.color_map = color_map
        
        # Проверка соответствия классов
        yolo_classes = set(self.yolo_model.names.values())
        layout_classes = set(self.layout_model.label_map.values())
        if yolo_classes != layout_classes:
            if self.noise_level > 0:
                print(f"Предупреждение: Несоответствие классов:")
                print(f"YOLO классы: {yolo_classes}")
                print(f"LayoutParser классы: {layout_classes}")
            raise ValueError(
                f"Несоответствие классов:\nYOLO: {yolo_classes}\nLayoutParser: {layout_classes}"
            )

    def _get_yolo_predictions(self, image: np.ndarray) -> List:
        """Получение предсказаний от YOLO модели."""
        results = self.yolo_model.predict(image, conf=self.yolo_model.conf, iou=self.yolo_model.iou)
        predictions = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = result.names[cls]
                
                if self.noise_level > 0:
                    print(f"YOLO обнаружил {class_name} с уверенностью {conf:.2f} в координатах ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                
                bbox = lp.elements.Rectangle(x1, y1, x2, y2)
                block = lp.elements.Layout([lp.elements.TextBlock(
                    block=bbox,
                    type=class_name,
                    score=float(conf)
                )])
                predictions.extend(block)
        
        return predictions

    def _get_layout_parser_predictions(self, image: np.ndarray) -> List:
        """Получение предсказаний от LayoutParser модели."""
        # Убеждаемся, что изображение в правильном формате
        if isinstance(image, torch.Tensor):
            if self.noise_level > 0:
                print("Преобразование PyTorch тензора в numpy array")
            image = image.cpu().numpy()
            if image.ndim == 4:
                image = image.squeeze(0)
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            
        if not isinstance(image, np.ndarray):
            raise ValueError("Изображение должно быть numpy array")
            
        if image.dtype != np.uint8:
            if self.noise_level > 0:
                print("Нормализация изображения в uint8")
            image = (image * 255).astype(np.uint8)
            
        # Проверяем и конвертируем в RGB если нужно
        if len(image.shape) == 2:
            if self.noise_level > 0:
                print("Преобразование grayscale изображения в RGB")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            if self.noise_level > 0:
                print("Изображение уже в формате RGB")
            pass  # RGB формат, всё хорошо
        else:
            raise ValueError("Неподдерживаемый формат изображения")
            
        # Получаем предсказания
        predictions = self.layout_model.detect(image)
        
        if self.noise_level > 0:
            print("\nПредсказания LayoutParser:")
            for pred in predictions:
                x1, y1, x2, y2 = pred.coordinates
                print(f"Обнаружен {pred.type} с уверенностью {pred.score:.2f} в координатах ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        return predictions

    def _calculate_iou(self, box1, box2) -> float:
        """Вычисление IoU между двумя боксами."""
        x1_1, y1_1, x2_1, y2_1 = box1.coordinates
        x1_2, y1_2, x2_2, y2_2 = box2.coordinates
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def _merge_predictions(self, layout1: List, layout2: List) -> List:
        """
        Объединение предсказаний двух моделей.
        
        Args:
            layout1: предсказания от первой модели
            layout2: предсказания от второй модели
            
        Returns:
            List: объединенные предсказания
        """
        def weighted_box_fusion(box1, box2, weight1, weight2):
            """Взвешенное объединение двух боксов."""
            x1_1, y1_1, x2_1, y2_1 = box1.coordinates
            x1_2, y1_2, x2_2, y2_2 = box2.coordinates
            
            total_weight = weight1 + weight2
            
            # Взвешенное среднее координат
            x1 = (x1_1 * weight1 + x1_2 * weight2) / total_weight
            y1 = (y1_1 * weight1 + y1_2 * weight2) / total_weight
            x2 = (x2_1 * weight1 + x2_2 * weight2) / total_weight
            y2 = (y2_1 * weight1 + y2_2 * weight2) / total_weight
            
            if self.noise_level > 0:
                print(f"Объединение боксов:")
                print(f"  Бокс 1: {box1.type} ({weight1:.2f}) - ({x1_1:.1f}, {y1_1:.1f}, {x2_1:.1f}, {y2_1:.1f})")
                print(f"  Бокс 2: {box2.type} ({weight2:.2f}) - ({x1_2:.1f}, {y1_2:.1f}, {x2_2:.1f}, {y2_2:.1f})")
                print(f"  Результат: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            return lp.elements.Rectangle(x1, y1, x2, y2)
        
        # Объединяем все боксы в один список для удобства
        all_boxes = layout1 + layout2
        all_boxes = sorted(all_boxes, key=lambda x: x.score, reverse=True)
        
        if self.noise_level > 0:
            print(f"\nНачало объединения предсказаний:")
            print(f"Всего боксов: {len(all_boxes)}")
        
        merged = []
        used_indices = set()
        
        # Проходим по всем боксам
        for i, box1 in enumerate(all_boxes):
            if i in used_indices:
                continue
                
            current_box = box1
            current_score = box1.score
            current_type = box1.type
            
            if self.noise_level > 0:
                print(f"\nОбработка бокса {i+1}: {current_type} ({current_score:.2f})")
            
            # Ищем все пересекающиеся боксы
            overlapping_boxes = []
            overlapping_scores = []
            overlapping_types = []
            
            for j, box2 in enumerate(all_boxes):
                if j == i or j in used_indices:
                    continue
                    
                iou = self._calculate_iou(current_box, box2)
                
                if self.noise_level > 0 and iou > 0:
                    print(f"  IoU с боксом {j+1} ({box2.type}): {iou:.2f}")
                
                # Используем более низкий порог IoU для объединения
                if iou > 0.3:  # Снизили порог с 0.5 до 0.3
                    overlapping_boxes.append(box2)
                    overlapping_scores.append(box2.score)
                    overlapping_types.append(box2.type)
                    used_indices.add(j)
            
            # Если есть пересекающиеся боксы
            if overlapping_boxes:
                if self.noise_level > 0:
                    print(f"  Найдено {len(overlapping_boxes)} пересекающихся боксов")
                
                # Объединяем все пересекающиеся боксы
                final_box = current_box
                final_score = current_score
                scores_sum = current_score
                
                for box, score in zip(overlapping_boxes, overlapping_scores):
                    final_box = weighted_box_fusion(
                        final_box, box,
                        final_score, score
                    )
                    scores_sum += score
                
                # Вычисляем средний score
                final_score = scores_sum / (len(overlapping_boxes) + 1)
                
                # Выбираем тип с наивысшим score
                all_types = [current_type] + overlapping_types
                all_scores = [current_score] + overlapping_scores
                final_type = all_types[all_scores.index(max(all_scores))]
                
                if self.noise_level > 0:
                    print(f"  Итоговый тип: {final_type}")
                    print(f"  Итоговый score: {final_score:.2f}")
                
                # Создаем новый блок
                block = lp.elements.Layout([lp.elements.TextBlock(
                    block=final_box,
                    type=final_type,
                    score=final_score
                )])
                merged.extend(block)
            else:
                # Если нет пересечений, добавляем исходный бокс
                if self.noise_level > 0:
                    print("  Нет пересекающихся боксов")
                merged.extend([box1])
            
            used_indices.add(i)
        
        if self.noise_level > 0:
            print(f"\nЗавершено объединение предсказаний. Итоговое количество боксов: {len(merged)}")
        
        return merged

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Обработка изображения."""
        # Загрузка и проверка изображения
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                if len(image.shape) != 3:
                    raise ValueError("Неверный формат изображения")
            else:
                raise ValueError("Неподдерживаемый формат изображения")
        elif isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:  # (B, C, H, W)
                image = image.squeeze(0)
            if image.dim() == 3:  # (C, H, W)
                image = image.permute(1, 2, 0).cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
        else:
            raise ValueError("Неподдерживаемый тип изображения")
        
        # Получаем предсказания от обеих моделей
        layout_predictions = self._get_layout_parser_predictions(image)
        yolo_predictions = self._get_yolo_predictions(image)
        
        # Объединяем предсказания
        merged_predictions = self._merge_predictions(layout_predictions, yolo_predictions)
        
        # Визуализируем результат
        result = lp.draw_box(image, merged_predictions, color_map=self.color_map, show_element_type=False)
        
        # Если результат уже PIL Image, используем его напрямую
        if isinstance(result, Image.Image):
            result_pil = result
        else:
            result_pil = Image.fromarray(result)
            
        # Добавляем подписи
        draw = ImageDraw.Draw(result_pil)
        for box in merged_predictions:
            x1, y1, x2, y2 = box.coordinates
            label_text = f"{box.type} ({box.score:.2f})"
            text_y = max(0, y1 - 15)
            draw.text((x1, text_y), label_text, fill=self.color_map[box.type])
        
        return result_pil 