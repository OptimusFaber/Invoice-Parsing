import cv2
import imutils
import pytesseract
from pytesseract import Output
from .text_processing import sort_words_into_columns, extract_text_from_image
import PIL
import numpy as np
import os

def process_image(image, debug=False, black_list=None, preprocess=True, 
                 y_threshold=30.0):
    """
    Обработка изображения: поворот, извлечение текста и сортировка по столбцам и строкам.
    
    Args:
        image_path: путь к изображению
        debug: флаг для включения отладочной информации
        black_list: список символов для удаления (если они встречаются как отдельные слова)
        preprocess: применять ли предварительную обработку изображения
        y_threshold: допустимая разница в y-координате для элементов на одной строке
        
    Returns:
        Tuple[List, List]: (столбцы с текстом и координатами, строки с текстом и координатами)
    """
    # Загружаем и поворачиваем изображение
    if type(image) == PIL.Image.Image:
        image = np.array(image)
    elif type(image) == np.ndarray:
        image = image
    else:
        try:
            path = os.path.join(image)
            image = cv2.imread(path)
        except:
            raise ValueError("Неверный тип изображения")
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rotated = imutils.rotate_bound(image, angle=results["rotate"])
    except:
        # Если не удалось определить ориентацию, используем исходное изображение
        rotated = image
    
    processed_img = rotated

    # Извлекаем текст с координатами
    words = extract_text_from_image(processed_img)
    
    # Фильтруем слова из черного списка
    if black_list:
        filtered_words = []
        for word in words:
            # Проверяем, не является ли слово одиночным символом из черного списка
            if not (len(word['text']) == 1 and word['text'] in black_list):
                filtered_words.append(word)
        
        if debug and len(filtered_words) < len(words):
            print(f"Удалено {len(words) - len(filtered_words)} слов из черного списка")
        
        words = filtered_words
    
    # Сортируем по столбцам с учетом минимального расстояния между столбцами
    columns = sort_words_into_columns(words, verbose=debug)
    
    # Формируем горизонтальные строки из слов на одной высоте
    # Сначала группируем слова по примерно одинаковой высоте
    text_lines = []
    all_words = []
    for column in columns:
        all_words.extend(column)
    
    # Сортируем все слова по y-координате (сверху вниз)
    all_words.sort(key=lambda w: w['bbox'][1])
    
    # Группируем слова по строкам (слова на одной высоте)
    current_line = []
    current_y = None
    
    for word in all_words:
        y = word['bbox'][1]
        
        if current_y is None:
            current_y = y
            current_line.append(word)
        elif abs(y - current_y) <= y_threshold:
            current_line.append(word)
        else:
            # Сортируем слова в строке по x-координате (слева направо)
            current_line.sort(key=lambda w: w['bbox'][0])
            text_lines.append(current_line)
            current_line = [word]
            current_y = y
    
    # Добавляем последнюю строку, если она не пустая
    if current_line:
        current_line.sort(key=lambda w: w['bbox'][0])
        text_lines.append(current_line)
    
    if debug:
        return columns, text_lines, get_debug_info(words, columns, text_lines, black_list, preprocess)
    return columns, text_lines


def get_debug_info(words, columns, text_lines, black_list=None, preprocess=False):
    """
    Формирует отладочную информацию о процессе обработки текста.
    
    Args:
        words: список слов с координатами
        columns: список столбцов
        text_lines: итоговые текстовые строки
        black_list: список символов для удаления
        preprocess: применялась ли предварительная обработка
        
    Returns:
        dict: словарь с отладочной информацией
    """
    debug_info = {
        "raw_words": words,
        "columns": columns,
        "text_lines": text_lines,
        "statistics": {
            "total_words": len(words),
            "total_columns": len(columns),
            "total_lines": len(text_lines)
        }
    }
    
    if black_list:
        debug_info["black_list"] = black_list
    
    if preprocess:
        debug_info["preprocessing"] = {
            "applied": True,
        }
    
    return debug_info