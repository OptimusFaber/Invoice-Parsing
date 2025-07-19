import random
import spacy
from pathlib import Path
import argparse
from tqdm import tqdm

def split_spacy_data(input_file, train_output, dev_output, split_ratio=0.8, seed=42):
    """
    Разделяет spaCy файл на тренировочную и валидационную выборки.
    
    Args:
        input_file (str): Путь к входному файлу spaCy
        train_output (str): Путь для сохранения тренировочного файла
        dev_output (str): Путь для сохранения валидационного файла
        split_ratio (float): Доля данных для тренировочной выборки (0.0-1.0)
        seed (int): Семя для воспроизводимости разделения
    """
    print(f"Загрузка данных из: {input_file}")
    
    # Загружаем данные из входного файла
    doc_bin = spacy.tokens.DocBin().from_disk(input_file)
    nlp = spacy.blank("nl")
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"Загружено {len(docs)} документов")
    
    # Перемешиваем документы с фиксированным сидом для воспроизводимости
    random.seed(seed)
    random.shuffle(docs)
    
    # Определяем точку разделения
    split_point = int(len(docs) * split_ratio)
    train_docs = docs[:split_point]
    dev_docs = docs[split_point:]
    
    print(f"Разделение: {len(train_docs)} документов для обучения, {len(dev_docs)} для валидации")
    
    # Создаем и сохраняем тренировочные данные
    train_doc_bin = spacy.tokens.DocBin()
    for doc in tqdm(train_docs, desc="Обработка тренировочных документов"):
        train_doc_bin.add(doc)
    train_doc_bin.to_disk(train_output)
    
    # Создаем и сохраняем валидационные данные
    dev_doc_bin = spacy.tokens.DocBin()
    for doc in tqdm(dev_docs, desc="Обработка валидационных документов"):
        dev_doc_bin.add(doc)
    dev_doc_bin.to_disk(dev_output)
    
    print(f"Готово! Данные сохранены в:")
    print(f"  - Тренировочные: {train_output}")
    print(f"  - Валидационные: {dev_output}")

def main():
    parser = argparse.ArgumentParser(description="Разделяет spaCy датасет на тренировочную и валидационную выборки")
    parser.add_argument("--input", "-i", required=True, help="Путь к входному файлу spaCy")
    parser.add_argument("--train", "-t", required=True, help="Путь для сохранения тренировочного файла")
    parser.add_argument("--dev", "-d", required=True, help="Путь для сохранения валидационного файла")
    parser.add_argument("--ratio", "-r", type=float, default=0.8, help="Доля данных для тренировочной выборки (по умолчанию: 0.8)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Семя для воспроизводимости разделения (по умолчанию: 42)")
    
    args = parser.parse_args()
    split_spacy_data(args.input, args.train, args.dev, args.ratio, args.seed)

if __name__ == "__main__":
    main() 