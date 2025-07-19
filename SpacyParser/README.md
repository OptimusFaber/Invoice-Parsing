## Training NER Model in spaCy

This project contains scripts for data preparation and training of Named Entity Recognition (NER) model using the spaCy library.

### Project Contents

- `json_to_spacy_converter.py` - script for converting JSON data to spaCy format
- `split_data.py` - script for splitting data into training and validation sets
- `config_fixed.cfg` - fixed configuration file for training spaCy model

### Steps for Training the Model

#### 1. Converting JSON to spaCy Format

First, you need to convert your labeled data from JSON to spaCy format:

```bash
python json_to_spacy_converter.py --input your_data.json --output all_data.spacy
```

Parameters:
- `--input, -i` - path to input JSON file
- `--output, -o` - path to save output spaCy file
- `--lang, -l` - model language (default: nl - Dutch)

#### 2. Splitting Data into Training and Validation Sets

Then split the data into training and validation sets:

```bash
python split_data.py --input all_data.spacy --train train_data.spacy --dev dev_data.spacy --ratio 0.8
```

Parameters:
- `--input, -i` - path to input spaCy file (all_data.spacy)
- `--train, -t` - path to save training file
- `--dev, -d` - path to save validation file
- `--ratio, -r` - training data ratio (default: 0.8)
- `--seed, -s` - fixed value for reproducibility (default: 42)

#### 3. Training the Model

After data preparation, you can start model training using the configuration file:

```bash
python -m spacy train config_fixed.cfg --output ./output --gpu-id 0
```

Paths to data are already specified in the configuration file:
```
[paths]
train = "./train_data.spacy"
dev = "./dev_data.spacy"
```

### Fixed Configuration

The project includes `config_fixed.cfg` file that contains fixes for the most common issues:

1. **Fixed language code**: uses correct code "nl" instead of "nld" for Dutch language
2. **Fixed length mismatch in MultiHashEmbed**: parameters `attrs` and `rows` have the same number of elements
3. **GPU usage configured**: parameter `gpu_allocator = "pytorch"` is set

### Difference Between Train and Dev Data

- **Training data (train)** - used for model training and weight adjustment
- **Validation data (dev)** - used for model quality evaluation during training, preventing overfitting (early stopping) and selecting the best model

### Main Configuration Settings

In the configuration file `config_fixed.cfg` you can modify the following key parameters:

- **Language**: `[nlp]` → `lang = "nl"` (use language code, e.g., "en" for English)
- **Batch size**: `[nlp]` → `batch_size = 128` (increase/decrease depending on GPU memory)
- **Dropout**: `[training]` → `dropout = 0.2` (increase if model is overfitting)
- **Maximum epochs**: `[training]` → `max_epochs = 20`
- **Learning rate**: `[training.optimizer]` → `learn_rate = 0.001`

### Troubleshooting Common Errors

#### Length Mismatch Error in MultiHashEmbed

If you see the error:
```
ValueError: Mismatched lengths: 2 vs 5
```

This means that `attrs` and `rows` lists in section `[components.tok2vec.model.embed]` have different number of elements. Make sure they have the same number of elements:

```
[components.tok2vec.model.embed]
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,1000,2500,2500]
```

#### Language Code Error

If you see the error:
```
ValueError: [E958] Language code defined in config (nld) does not match language code of current Language subclass Dutch (nl)
```

Make sure you're using the correct language code in configuration. For Dutch language it's "nl", not "nld".

### Using the Trained Model

After training, the model will be saved in the specified `--output` directory. You can load and use it as follows:

```python
import spacy

# Load the model
nlp = spacy.load("./output/model-best")

# Process text
text = "Your text for analysis"
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
``` 