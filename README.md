# Detecting Data Leakage in Deep Learning Using CNN and Vision Transformer Representations

**Name:** Bhavya Lahari Vaddempudi

---

## Project Overview

This project focuses on detecting data leakage in deep learning datasets. Data leakage happens when similar or identical samples appear in both training and test sets, which leads to misleadingly high accuracy.

The system uses deep learning representations from CNN and Vision Transformer models to identify both exact duplicates and near-duplicate samples. A similarity-based detection approach is used to classify whether leakage exists.

---

## Key Idea

Instead of comparing raw images directly, this project:

* Extracts feature representations using CNN and Vision Transformer
* Computes similarity between train and test samples
* Uses a small neural network to detect leakage

This allows detection of subtle leakage cases where images are slightly modified.

---

## Dataset

* Dataset used: Fashion-MNIST
* Total classes: 10

Three dataset versions are created:

* Clean dataset (no leakage)
* Exact leakage (same images in train and test)
* Near leakage (images with small transformations like rotation, flip, noise)

---

## Models Used

1. CNN-based feature extractor
2. Vision Transformer (ViT) feature extractor
3. MLP-based leakage detector
4. Baseline method using simple hash comparison

---

## Project Structure

```
data_builder/
    leak_creator.py

rep_extractors/
    conv_extractor.py
    transformer_extractor.py

leak_detector/
    detector_head.py
    sim_calculator.py

runner_scripts/
    create_leak_versions.py
    pull_representations.py
    train_detector_conv.py
    train_detector_transformer.py
    simple_hash_baseline.py

main.py
config.yaml
utils.py
requirements.txt
README.md
```

---

## How to Run the Project

Step 1: Install dependencies

```
pip install -r requirements.txt
```

Step 2: Create dataset with leakage

```
python runner_scripts/create_leak_versions.py
```

Step 3: Extract representations

```
python runner_scripts/pull_representations.py
```

Step 4: Train CNN-based detector

```
python runner_scripts/train_detector_conv.py
```

Step 5: Train Transformer-based detector

```
python runner_scripts/train_detector_transformer.py
```

Step 6: Run baseline method

```
python runner_scripts/simple_hash_baseline.py
```

Step 7: Run full pipeline

```
python main.py
```

---

## Outputs Generated

After running the project, the following outputs will be created:

* Confusion matrices for both models
* Training and validation accuracy graphs
* Model comparison chart
* Performance metrics (accuracy, precision, recall, F1-score)

---

## Evaluation Metrics

The system is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

These metrics help measure how well the system detects leakage.

---

## Results Summary

* Baseline method works only for exact duplicates
* CNN model improves detection using learned features
* Vision Transformer performs best, especially for near-duplicate cases

---

## Limitations

* May produce false positives for very similar images
* Performance depends on quality of feature extraction
* Dataset is controlled and may not fully reflect real-world scenarios

---

## Reproducibility

This project is fully reproducible:

* All code is included
* No hidden steps
* Clear execution commands provided
* Same results can be reproduced using the given scripts

---

## Requirements

See `requirements.txt` for full dependency list.

---

## Final Note

This project demonstrates how deep learning can be used not just for prediction tasks, but also for improving data quality and ensuring reliable model evaluation.

