import os
import requests
from PIL import Image
import spacy
import json
import cv2
import imutils
from tqdm import tqdm
from TextParser.parser import process_image

# === Настройки ===
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjUwMWM5NGFkLTNlZmItNGUwZC04NWQ5LTIzNjY2ZDU0MTRjOSJ9.v02ARUFBqcgxH6V60JbzY7FS8p9zUdSdZA0rLcDlA5E"
OLLAMA_URL = "http://92.111.200.91:5000/api/chat/completions"
IMAGE_DIR = "/home/rodrick/Desktop/My script/Invoice_detail/train/images"
OUTPUT_FILE = "/home/rodrick/Desktop/My script/train_data.spacy"

# === Загружаем пустую модель spaCy ===
nlp = spacy.blank("nld")

def classify_text_lines(text_lines):
    """
    Классифицирует строки текста с помощью модели языка
    
    Args:
        text_lines: список строк текста
        
    Returns:
        словарь с классифицированными сущностями
    """
    # Формируем текст для классификации
    blocks_text = "\n".join(
        f'Line: "{line}"' 
        for i, line in enumerate(text_lines)
    )

    prompt = """Classify each line into categories: ClientId, ClientVAT, InvoiceDate, InvoiceNumber or "None".
Take the whole line if it contains a value with its label.

Example:
Line: "Datum 5 november 2007" -> InvoiceDate
Line: "Debiteur nr. 429" -> ClientId
Line: "Factuurnummer 28007494" -> InvoiceNumber

Input:
{}

Return JSON array with classifications in format:
[{{"text": "full line text", "label": "category"}}]""".format(blocks_text)

    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen2.5:latest",
        "messages": [
            {
                "role": "system", 
                "content": "You are a precise invoice text classifier. Always return complete, valid JSON with all lines classified."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Clean up response
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1]
        if "```" in content:
            content = content.split("```")[0]
        content = content.strip()
        
        try:
            classifications = json.loads(content)
            
            # Преобразуем классификации в словарь сущностей
            entities = {}
            for classification in classifications:
                if classification["label"] != "None":
                    entities[classification["label"]] = classification["text"]
            
            return entities
            
        except json.JSONDecodeError as e:
            tqdm.write(f"Error parsing JSON response: {e}")
            tqdm.write("Raw response:", content[:200] + "..." if len(content) > 200 else content)
            return {}
            
    except requests.RequestException as e:
        tqdm.write(f"Error in request to Open WebUI: {e}")
        return {}

def convert_to_spacy_format(json_data):
    """
    Преобразует JSON-данные в формат spaCy
    
    Args:
        json_data: список словарей с полями text и entities
        
    Returns:
        DocBin: объект DocBin с документами spaCy
    """
    db = spacy.tokens.DocBin()
    
    for example in json_data:
        text = example["text"]
        doc = nlp(text)
        ents = []
        
        for label, value in example["entities"].items():
            # Находим начало и конец сущности в тексте
            start_index = text.find(value)
            if start_index != -1:
                end_index = start_index + len(value)
                # Создаем Span из символьных позиций
                span = doc.char_span(start_index, end_index, label=label)
                if span is not None:
                    ents.append(span)
                else:
                    tqdm.write(f"Не удалось создать Span для '{value}' в тексте")
        
        doc.ents = ents
        db.add(doc)
    
    return db

def main():
    json_data = []
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp"))]
    
    # Создаем progress bar
    for filename in tqdm(files, desc="Обработка изображений", unit="img"):
        path = os.path.join(IMAGE_DIR, filename)
        
        try:
            # Используем TextParser.parser.process_image для извлечения текста
            _, text_lines = process_image(path)
            
            # Преобразуем в простой список строк
            strings = []
            for line in text_lines:
                line_text = " ".join(block["text"] for block in line)
                strings.append(line_text)
            
            if not strings:
                tqdm.write(f"Пропуск {filename}: не удалось распознать текст")
                continue
            
            # Печатаем результат OCR
            tqdm.write("\nРезультат OCR по строкам:")
            for i, line_text in enumerate(strings):
                tqdm.write(f"Строка {i+1}: {line_text}")
            
            # Получаем классификацию
            entities = classify_text_lines(strings)
            if not entities:
                tqdm.write(f"Пропуск {filename}: не удалось классифицировать строки")
                continue
            
            # Формируем полный текст
            full_text = " ".join(strings)
            
            # Создаем JSON элемент
            json_item = {
                "text": full_text,
                "entities": entities
            }
            
            json_data.append(json_item)
            
            # Печатаем результат конвертации
            tqdm.write("\nСохраненный результат:")
            tqdm.write(f"Text: {json_item['text']}")
            tqdm.write("Entities:")
            for label, value in json_item['entities'].items():
                tqdm.write(f"  {label}: {value}")
            tqdm.write("-------------------")
            
            # Сохраняем промежуточный результат каждые 5 файлов
            if len(json_data) % 5 == 0:
                with open(OUTPUT_FILE.replace('.spacy', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                tqdm.write(f"✓ Промежуточное сохранение JSON: обработано {len(json_data)} файлов")
            
        except Exception as e:
            tqdm.write(f"Ошибка при обработке {filename}: {str(e)}")
            continue

    # Финальное сохранение JSON
    if json_data:
        json_path = OUTPUT_FILE.replace('.spacy', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Конвертируем JSON в spaCy формат
        db = convert_to_spacy_format(json_data)
        
        # Сохраняем spaCy датасет
        db.to_disk(OUTPUT_FILE)
        print(f"\n✅ Данные сохранены в:")
        print(f"JSON: {json_path}")
        print(f"spaCy: {OUTPUT_FILE}")
        print(f"Всего обработано: {len(json_data)} файлов")
    else:
        print("\n❌ Не удалось обработать ни одного файла")

if __name__ == "__main__":
    main()