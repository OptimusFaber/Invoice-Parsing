#!/bin/bash

# Проверка существования директории output
if [ ! -d "./output" ]; then
    mkdir -p ./output
fi

# Запуск обучения модели с улучшенной конфигурацией
echo "🚀 Запуск обучения модели с улучшенной конфигурацией..."
python -m spacy train "./new config.cfg" --output ./output --gpu-id 0

# После завершения обучения проверка результатов
if [ -f "./output/model-best/meta.json" ]; then
    echo "✅ Обучение завершено успешно!"
    echo "📊 Результаты обучения (лучшая модель):"
    python -m spacy evaluate ./output/model-best ./dev_data.spacy
else
    echo "❌ Что-то пошло не так, модель не найдена."
fi 