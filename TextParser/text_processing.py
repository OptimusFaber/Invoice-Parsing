import numpy as np
from typing import List, Dict, Union
from PIL import Image
import pytesseract

def merge_nearby_words(
    words: List[Dict[str, Union[str, List[float]]]],
    max_word_gap: float = None,
    verbose: bool = False
) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Объединение близких слов в строках.
    
    Args:
        words: список словарей слов с ключами 'text', 'bbox' (x, y, width, height)
        max_word_gap: максимальное расстояние между правым краем одного слова и левым краем другого
                 (если None, то вычисляется как средняя ширина символа * 2)
        verbose: выводить ли подробную информацию
    """
    if not words:
        return []
    
    # Сортируем слова по y-координате нижнего края
    words_sorted = sorted(words, key=lambda w: w['bbox'][1] + w['bbox'][3])
    
    # Группируем слова по строкам, используя нижний край
    rows = []
    current_row = [words_sorted[0]]
    # Используем y-координату нижнего края первого слова
    current_bottom_y = words_sorted[0]['bbox'][1] + words_sorted[0]['bbox'][3]
    
    for word in words_sorted[1:]:
        y = word['bbox'][1]
        h = word['bbox'][3]
        bottom_y = y + h
        y_diff = abs(bottom_y - current_bottom_y)
        
        # Используем меньший порог для разницы по y, так как теперь сравниваем нижние края
        if y_diff < h * 0.3:  # Уменьшили порог с 0.5 до 0.3
            current_row.append(word)
        else:  # Новая строка
            # Сортируем слова в текущей строке по x-координате
            current_row.sort(key=lambda w: w['bbox'][0])
            rows.append(current_row)
            current_row = [word]
            current_bottom_y = bottom_y
    
    # Добавляем последнюю строку
    if current_row:
        current_row.sort(key=lambda w: w['bbox'][0])
        rows.append(current_row)
    
    if verbose:
        print("\nСгруппированные строки до объединения:")
        for i, row in enumerate(rows):
            print(f"Строка {i + 1}:", " | ".join(word['text'] for word in row))
    
    # Объединяем близкие слова в каждой строке
    merged_words = []
    for row in rows:
        # Продолжаем объединять слова, пока есть что объединять
        while len(row) > 1:
            merged_something = False
            for i in range(len(row) - 1):
                current_word = row[i]
                next_word = row[i + 1]
                
                if current_word is None or next_word is None:
                    continue
                    
                x1, y1, w1, h1 = current_word['bbox']
                x2, y2, w2, h2 = next_word['bbox']
                x_gap = x2 - (x1 + w1)
                
                if x_gap <= max_word_gap:  # Слова достаточно близко
                    new_text = current_word['text'] + " " + next_word['text']
                    new_width = (x2 + w2) - x1
                    merged_word = {
                        'text': new_text,
                        'bbox': [x1, min(y1, y2), new_width, max(h1, h2)]
                    }
                    if verbose:
                        print(f"Объединяем слова: '{current_word['text']}' + '{next_word['text']}' = '{new_text}'")
                    
                    # Заменяем первое слово объединенным, а второе помечаем как None
                    row[i] = merged_word
                    row[i + 1] = None
                    merged_something = True
            
            # Убираем None из списка
            row = [word for word in row if word is not None]
            
            # Если ничего не объединили за проход, выходим из цикла
            if not merged_something:
                break
        
        merged_words.extend(row)
        
        if verbose:
            print(f"\nСтрока после объединения:", " | ".join(word['text'] for word in row))
    
    return merged_words

def range_subset(range1, range2):
    """Whether range1 is a subset of range2."""
    if not range1: return True  # empty range is subset of anything
    if not range2: return False  # non-empty range can't be subset of empty range
    if len(range1) > 1 and range1.step % range2.step:
        return False  # must have a single value or integer multiple step
    return range1.start in range2 and range1[-1] in range2

def sort_words_into_columns(
    words: List[Dict[str, Union[str, List[float]]]],
    max_x_distance: float = 30.0,
    max_word_gap: float = None,
    verbose: bool = False
) -> List[List[Dict[str, Union[str, List[float]]]]]:
    """
    Сортировка слов по столбцам на основе их левого верхнего угла.
    
    Args:
        words: список словарей слов с ключами 'text', 'bbox' (x, y, width, height)
        max_x_distance: максимальное расстояние по X между левыми краями слов в столбце
        max_word_gap: максимальное расстояние между словами для их объединения
                      (если None, то вычисляется как средняя ширина символа * 2)
        verbose: выводить ли подробную информацию
    """

    # Вычисляем среднюю ширину символа, если max_word_gap не задан
    if max_word_gap is None:
        total_width = 0
        total_chars = 0
        
        for word in words:
            text = word['text']
            if text and len(text) > 0:
                bbox = word['bbox']
                width = bbox[2]  # ширина bbox
                total_width += width
                total_chars += len(text)
        
        if total_chars > 0:
            avg_char_width = total_width / total_chars
            max_word_gap = int(1.5 * avg_char_width)
            if verbose:
                print(f"Средняя ширина символа: {avg_char_width:.2f}, max_word_gap: {max_word_gap:.2f}")
        else:
            max_word_gap = 15.0  # значение по умолчанию

    # Сначала объединяем близкие слова
    merged_words = merge_nearby_words(words, max_word_gap=max_word_gap, verbose=verbose)
    
    if not merged_words:
        return []
    
    # Вычисляем среднюю ширину символа для использования в качестве допустимого отклонения
    total_width = 0
    total_chars = 0
    
    for word in merged_words:
        text = word['text']
        if text and len(text) > 0:
            bbox = word['bbox']
            width = bbox[2]  # ширина bbox
            total_width += width
            total_chars += len(text)
    
    if total_chars == 0:
        avg_char_width = 10.0  # значение по умолчанию
    else:
        avg_char_width = total_width / total_chars
    
    if verbose:
        print(f"Средняя ширина символа для столбцов: {avg_char_width:.2f}")
        
    # Используем левый верхний угол вместо центра
    word_positions = []
    for word in merged_words:
        x, y, w, h = word['bbox']
        word_positions.append({
            'text': word['text'],
            'bbox': word['bbox'],
            'left_xy': (x, y),
            'right_x': x + w,  # добавляем X координату правого края
            'right_y': y + h  # добавляем Y координату правого края
        })
    
    # Сортируем слова по вертикали (сверху вниз)
    word_positions.sort(key=lambda w: w['left_xy'][1])
    
    # Создаем начальные столбцы
    columns = []
    
    # Для каждого слова находим подходящий столбец или создаем новый
    for word in word_positions:
        x_left = word['left_xy'][0]
        x_right = word['right_x']
        y = word['left_xy'][1]
        
        found_column = False
        
        for column in columns:
            # Проверяем все слова в столбце для определения диапазона столбца
            column_left_edges = [w['left_xy'][0] for w in column]
            column_right_edges = [w['right_x'] for w in column]
            
            # Вычисляем границы столбца
            # col_left = min(column_left_edges)
            # col_right = max(column_right_edges)
            col_left = column_left_edges[-1]
            col_right = column_right_edges[-1]
                
            # Правило 1: Левый верхний угол совпадает (с допустимым отклонением)
            left_aligned = abs(x_left - col_left) <= max_x_distance
            
            # Правило 2: Правый верхний угол совпадает (с допустимым отклонением)
            right_aligned = abs(x_right - col_right) <= max_x_distance
            
            # Правило 3: Слово находится между левым и правым краями столбца с учетом допустимого отклонения
            between_edges = range_subset(range(x_left, x_right), range(col_left-max_word_gap, col_right+max_word_gap))

            
            if left_aligned or right_aligned or between_edges:
                # Проверяем, что в столбце под каждым заголовком не более одного значения                
                column.append(word)
                found_column = True
                break
        
        # Если не нашли подходящий столбец, создаем новый
        if not found_column:
            columns.append([word])
    
    # Сортируем слова внутри каждого столбца по вертикали
    for column in columns:
        column.sort(key=lambda w: w['left_xy'][1])
    
    # Удаляем пустые столбцы
    columns = [col for col in columns if col]
    
    # Сортируем столбцы по горизонтали (слева направо)
    columns.sort(key=lambda col: min(w['left_xy'][0] for w in col))
    
    if verbose:
        print("\nРезультаты сортировки по столбцам:")
        for i, column in enumerate(columns):
            print(f"\nСтолбец {i+1}:")
            x_positions = [w['left_xy'][0] for w in column]
            print(f"Среднее положение по X: {np.mean(x_positions):.2f}")
            print(f"Стандартное отклонение: {np.std(x_positions):.2f}")
            print("Слова:")
            for word in column:
                print(f"  {word['text']} (позиция: {word['left_xy']}, правый край: {word['right_x']})")
    
    # Возвращаем столбцы в нужном формате
    return [[{
        'text': word['text'],
        'bbox': word['bbox']
    } for word in column] for column in columns]

def extract_text_from_image(
    image: Union[np.ndarray, Image.Image],
    x1: int = 0,
    y1: int = 0,
    x2: int = None,
    y2: int = None
) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Извлечение текста из изображения или его области.
    
    Args:
        image: исходное изображение
        x1, y1, x2, y2: координаты области для извлечения (если не указаны, используется всё изображение)
        
    Returns:
        List[Dict]: список слов с их координатами
    """
    # Преобразуем изображение если нужно
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
        
    # Если координаты не указаны, берем всё изображение
    if x2 is None:
        x2 = pil_image.width
    if y2 is None:
        y2 = pil_image.height
        
    # Вырезаем область
    region = pil_image.crop((x1, y1, x2, y2))
    
    # Получаем данные OCR
    data = pytesseract.image_to_data(region, output_type=pytesseract.Output.DICT)
    
    words = []
    # Собираем слова с их координатами
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # Пропускаем пустые строки
            # Корректируем координаты относительно исходного изображения
            x = data['left'][i] + x1
            y = data['top'][i] + y1
            w = data['width'][i]
            h = data['height'][i]
            
            words.append({
                'text': text,
                'bbox': [x, y, w, h]
            })
            
    return words 