from .invoice_processing import Parser
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from ultralytics import YOLO
import logging
import warnings
import json

# Настройка логирования
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DocumentParser")

# Перенаправление предупреждений detectron2 в логи
warnings.filterwarnings("ignore", category=UserWarning)

class TwoStageParser:
    def __init__(self, cuda: bool = False, noise_level: int = 0):
        """
        Инициализация двухэтапного парсера.
        
        Args:
            cuda: использовать ли CUDA
            noise_level: уровень вывода (0 - тихий, 1 - подробный)
        """
        self.cuda = cuda
        self.noise_level = noise_level
        self.first_stage_parser = None
        self.second_stage_parser = None
        
        # Настройка вывода YOLO
        if self.noise_level == 0:
            YOLO.args = dict(verbose=False)
            
        # Определяем корневую директорию DocumentParser
        self.root_dir = Path(__file__).parent
            
        # Загрузка конфигурации
        config_path = self.root_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def _resolve_path(self, relative_path: str) -> str:
        """
        Преобразует относительный путь в абсолютный относительно корня DocumentParser.
        
        Args:
            relative_path: относительный путь из конфигурационного файла
            
        Returns:
            str: абсолютный путь к файлу
        """
        absolute_path = self.root_dir / relative_path
        if not absolute_path.exists():
            raise FileNotFoundError(f"Файл не найден: {absolute_path}")
        return str(absolute_path)

    def initialize_first_stage(
        self,
        layout_config_path: str,
        layout_model_path: str,
        yolo_model_path: str,
        label_map: Dict[int, str],
        color_map: Dict[str, str]
    ) -> None:
        """
        Инициализация первого этапа - поиск основных областей документа.
        """
        self.first_stage_parser = Parser(
            layout_config_path=layout_config_path,
            layout_model_path=layout_model_path,
            yolo_model_path=yolo_model_path,
            label_map=label_map,
            color_map=color_map,
            cuda=self.cuda,
            noise_level=self.noise_level
        )
    
    def initialize_second_stage(
        self,
        layout_config_path: str,
        layout_model_path: str,
        yolo_model_path: str,
        label_map: Dict[int, str],
        color_map: Dict[str, str]
    ) -> None:
        """
        Инициализация второго этапа - поиск деталей в вырезанных областях.
        """
        self.second_stage_parser = Parser(
            layout_config_path=layout_config_path,
            layout_model_path=layout_model_path,
            yolo_model_path=yolo_model_path,
            label_map=label_map,
            color_map=color_map,
            cuda=self.cuda,
            noise_level=self.noise_level
        )

    def detect_document_areas(self, image: np.ndarray) -> List:
        """
        Первый этап: обнаружение основных областей документа.
        
        Args:
            image: изображение для анализа
            
        Returns:
            List: список обнаруженных областей
        """
        if self.first_stage_parser is None:
            raise ValueError("Первый этап не инициализирован. Вызовите initialize_first_stage()")
            
        predictions = self.first_stage_parser._merge_predictions(
            self.first_stage_parser._get_layout_parser_predictions(image),
            self.first_stage_parser._get_yolo_predictions(image)
        )
        return predictions

    def crop_areas(
        self,
        image: Union[np.ndarray, Image.Image],
        predictions: List,
        target_classes: List[str]
    ) -> List[Tuple[Union[np.ndarray, Image.Image], str, Tuple[int, int, int, int]]]:
        """
        Вырезание указанных областей из изображения.
        
        Args:
            image: исходное изображение
            predictions: список предсказанных областей от модели
            target_classes: список классов для вырезания
            
        Returns:
            List[Tuple]: список кортежей (вырезанная_область, класс, координаты)
        """
        # Преобразуем numpy array в PIL Image если нужно
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        crops = []
        for pred in predictions:
            # Получаем координаты и тип из предсказания
            if hasattr(pred, 'coordinates'):
                coords = pred.coordinates
                pred_type = pred.type
            else:
                # Для случая, когда предсказание - это словарь или другой объект
                try:
                    coords = pred['coordinates'] if isinstance(pred, dict) else pred[0]
                    pred_type = pred['type'] if isinstance(pred, dict) else pred[1]
                except (KeyError, IndexError, TypeError):
                    print(f"Пропускаем предсказание: невозможно извлечь координаты или тип")
                    continue
            
            # Проверяем, что тип в целевых классах
            if pred_type in target_classes:
                try:
                    # Преобразуем координаты в целые числа
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Проверяем корректность координат
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                        print(f"Пропускаем предсказание: некорректные координаты {(x1, y1, x2, y2)}")
                        continue
                        
                    # Обрезаем изображение
                    crop = pil_image.crop((x1, y1, x2, y2))
                    crops.append((crop, pred_type, (x1, y1, x2, y2)))
                except (ValueError, TypeError) as e:
                    print(f"Ошибка при обработке координат: {e}")
                    continue
                
        return crops

    def detect_details_in_crops(
        self,
        crops: List[Tuple[np.ndarray, str, Tuple[int, int, int, int]]]
    ) -> List[Tuple[Image.Image, str, Tuple[int, int, int, int]]]:
        """
        Третий этап: поиск деталей в вырезанных областях.
        
        Args:
            crops: список кортежей (вырезанная_область, класс, координаты)
            
        Returns:
            List[Tuple]: список кортежей (результат_анализа, класс, координаты)
        """
        if self.second_stage_parser is None:
            raise ValueError("Второй этап не инициализирован. Вызовите initialize_second_stage()")
            
        results = []
        for crop, crop_type, coords in crops:
            result = self.second_stage_parser.predict(crop)
            results.append((result, crop_type, coords))
        return results

    def visualize_results(
        self,
        image: np.ndarray,
        first_stage_result: Image.Image,
        detailed_results: List[Tuple[Image.Image, str, Tuple[int, int, int, int]]]
    ) -> Image.Image:
        """
        Визуализация результатов анализа.
        
        Args:
            image: исходное изображение
            first_stage_result: результат первого этапа
            detailed_results: результаты анализа вырезанных областей
            
        Returns:
            Image.Image: итоговое изображение с визуализацией
        """
        # Получаем размеры исходного изображения
        first_width, first_height = first_stage_result.size
        
        # Создаем белый фон для правой части
        right_image = Image.new('RGB', (first_width, first_height), 'white')
        
        # Вычисляем размеры для каждой вырезанной области
        n_crops = len(detailed_results)
        if n_crops > 0:
            # Определяем максимальное количество областей в строке
            max_cols = min(3, n_crops)
            n_rows = (n_crops + max_cols - 1) // max_cols
            
            # Вычисляем размеры для каждой области
            cell_width = first_width // max_cols
            cell_height = first_height // n_rows
            
            # Размещаем каждую область
            for i, (result, crop_type, _) in enumerate(detailed_results):
                # Определяем позицию в сетке
                row = i // max_cols
                col = i % max_cols
                
                # Масштабируем результат под размер ячейки
                result_resized = result.resize((cell_width - 20, cell_height - 40))
                
                # Вычисляем координаты для вставки
                x_offset = col * cell_width + 10
                y_offset = row * cell_height + 10
                
                # Добавляем подпись
                draw = ImageDraw.Draw(right_image)
                draw.text((x_offset, y_offset - 20), f"Crop type: {crop_type}", fill="black")
                
                # Вставляем изображение
                right_image.paste(result_resized, (x_offset, y_offset))
        
        # Создаем финальное изображение
        total_width = first_width * 2
        result_image = Image.new('RGB', (total_width, first_height))
        result_image.paste(first_stage_result, (0, 0))
        result_image.paste(right_image, (first_width, 0))
        
        return result_image

    def process_single_stage(
        self,
        image: np.ndarray,
        stage: str = "first"
    ) -> Tuple[Image.Image, List]:
        """
        Обработка изображения в одну стадию.
        
        Args:
            image: изображение для анализа
            stage: какую стадию использовать ("first" или "second")
            
        Returns:
            Tuple[Image.Image, List]: результат обработки и список предсказаний
        """
        if stage == "first":
            if self.first_stage_parser is None:
                raise ValueError("Первый этап не инициализирован")
            predictions = self.detect_document_areas(image)
            result = self.first_stage_parser.predict(image)
        else:
            if self.second_stage_parser is None:
                raise ValueError("Второй этап не инициализирован")
            predictions = self.second_stage_parser._merge_predictions(
                self.second_stage_parser._get_layout_parser_predictions(image),
                self.second_stage_parser._get_yolo_predictions(image)
            )
            result = self.second_stage_parser.predict(image)
            
        return result, predictions

    def process_image(
        self,
        image_path: str,
        mode: str = "two_stage",
        target_classes: Optional[List[str]] = None,
        stage: str = "first"
    ) -> Union[Tuple[Image.Image, List], Tuple[Image.Image, List, List[Tuple[Image.Image, str, Tuple[int, int, int, int]]]]]:
        """
        Обработка изображения с выбором режима.
        
        Args:
            image_path: путь к изображению
            mode: режим обработки ("single_stage" или "two_stage")
            target_classes: классы для вырезания (для two_stage режима)
            stage: какую стадию использовать в single_stage режиме ("first" или "second")
            
        Returns:
            Union[Tuple[Image.Image, List], Tuple[Image.Image, List, List[Tuple[Image.Image, str, Tuple[int, int, int, int]]]]]:
                - для single_stage: (визуализация, предсказания)
                - для two_stage: (визуализация, предсказания_первого_этапа, результаты_деталей)
        """
        # Проверка инициализации
        if self.first_stage_parser is None or self.second_stage_parser is None:
            self.initialize_default_stages()
        
        # Проверка существования файла
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")
            
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if mode == "single_stage":
            result, predictions = self.process_single_stage(image, stage)
            return result, predictions
        
        # Two-stage режим
        if target_classes is None:
            target_classes = ["Invoice_detail"]
            
        # Этап 1: Обнаружение областей
        first_stage_predictions = self.detect_document_areas(image)
        first_stage_result = self.first_stage_parser.predict(image)
        
        # Этап 2: Вырезание указанных областей
        crops = self.crop_areas(image, first_stage_predictions, target_classes)
        
        if not crops:
            print(f"Не найдено областей следующих классов: {target_classes}")
            return first_stage_result, first_stage_predictions, []
        
        # Этап 3: Анализ вырезанных областей
        detailed_results = self.detect_details_in_crops(crops)
        
        # Визуализация результатов
        final_visualization = self.visualize_results(image, first_stage_result, detailed_results)
        
        return final_visualization, first_stage_predictions, detailed_results

    def initialize_default_stages(self):
        """Инициализация стандартных настроек для обоих этапов."""
        # Первый этап
        first_stage = self.config["first_stage"]
        self.initialize_first_stage(
            layout_config_path=self._resolve_path(first_stage["layout"]["config_path"]),
            layout_model_path=self._resolve_path(first_stage["layout"]["model_path"]),
            yolo_model_path=self._resolve_path(first_stage["yolo"]["model_path"]),
            label_map={int(k): v for k, v in first_stage["label_map"].items()},
            color_map=first_stage["color_map"]
        )
        
        # Второй этап
        second_stage = self.config["second_stage"]
        self.initialize_second_stage(
            layout_config_path=self._resolve_path(second_stage["layout"]["config_path"]),
            layout_model_path=self._resolve_path(second_stage["layout"]["model_path"]),
            yolo_model_path=self._resolve_path(second_stage["yolo"]["model_path"]),
            label_map={int(k): v for k, v in second_stage["label_map"].items()},
            color_map=second_stage["color_map"]
        )
