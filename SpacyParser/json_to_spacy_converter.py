import json
import spacy
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_json_to_spacy(input_json, output_spacy, lang="nl"):
    """
    Конвертирует JSON файл в формат spaCy для NER обучения
    
    Args:
        input_json (str): путь к входному JSON файлу
        output_spacy (str): путь к выходному файлу spaCy
        lang (str): язык модели (по умолчанию "nl" - голландский)
    """
    # Загружаем пустую модель spaCy для указанного языка
    print(f"Загрузка пустой модели spaCy для языка: {lang}")
    nlp = spacy.blank(lang)
    
    # Загружаем JSON данные
    print(f"Чтение JSON файла: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Создаем DocBin для хранения обработанных документов
    doc_bin = spacy.tokens.DocBin()
    
    # Обрабатываем каждый пример
    print("Конвертация примеров...")
    for example in tqdm(data, desc="Обработка документов"):
        text = example["text"]
        entities = example.get("entities", {})
        
        # Создаем новый документ
        doc = nlp(text)
        
        # Добавляем сущности
        ents = []
        for label, value in entities.items():
            # Находим все вхождения значения в тексте
            start_index = text.find(value)
            if start_index != -1:
                end_index = start_index + len(value)
                # Создаем спан из символьных позиций
                span = doc.char_span(start_index, end_index, label=label)
                if span is not None:
                    ents.append(span)
                else:
                    tqdm.write(f"Предупреждение: не удалось создать спан для '{value}' (метка: {label})")
        
        # Устанавливаем сущности в документе
        doc.ents = ents
        
        # Добавляем документ в DocBin
        doc_bin.add(doc)
    
    # Сохраняем в файл
    output_path = Path(output_spacy)
    print(f"Сохранение {len(data)} документов в: {output_path}")
    doc_bin.to_disk(output_path)
    print("Конвертация завершена!")

def main():
    parser = argparse.ArgumentParser(description="Конвертирует JSON в формат spaCy для обучения NER")
    parser.add_argument("--input", "-i", required=True, help="Путь к входному JSON файлу")
    parser.add_argument("--output", "-o", required=True, help="Путь к выходному файлу spaCy")
    parser.add_argument("--lang", "-l", default="nld", help="Язык модели (по умолчанию: nld)")
    
    args = parser.parse_args()
    convert_json_to_spacy(args.input, args.output, args.lang)

if __name__ == "__main__":
    main() 