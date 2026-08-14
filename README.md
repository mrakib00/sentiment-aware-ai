# Flower Classification with MobileNetV2 Transfer Learning

**Author:** MD Rakib Hossain | **Student ID:** A00057300
**Module:** Machine Learning (CMP-L015-0), University of Roehampton
**Dataset:** [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

A transfer-learning image classifier that sorts flower photos into 102 species using a frozen MobileNetV2 backbone with a small custom classifier head. The project also studies where a frozen backbone reaches its limits on fine-grained, class-imbalanced data.

## What This Project Does

The Oxford 102 dataset contains 8,189 flower images across 102 species, many of which differ only in subtle texture or lighting rather than overall shape. This project loads that raw dataset, builds a clean class-wise folder structure, and trains a MobileNetV2 model (pretrained on ImageNet, backbone frozen) to classify the images. It then evaluates how well the frozen features cope with fine-grained differences and where they break down.

## Pipeline

1. **Data:** download the Oxford 102 images and `imagelabels.mat`, then convert the MATLAB 1-based labels to 0-based.
2. **Split:** a fixed 80/10/10 train/validation/test split (via `train_test_split`, `random_state=42`), with images copied into per-class folders so filenames and categories stay aligned.
3. **Preprocessing:** `ImageDataGenerator(rescale=1./255)`, images resized to 224x224, batch size 32, one-hot (categorical) labels. No augmentation was used on purpose, so any weakness could be traced to the frozen backbone rather than to added variety.
4. **Model:** MobileNetV2 (ImageNet weights, `include_top=False`, `trainable=False`), then GlobalAveragePooling2D, Dense(256, ReLU), Dropout(0.3), and Dense(102, softmax).
5. **Training:** Adam optimiser, categorical cross-entropy loss, 10 epochs.
6. **Evaluation:** accuracy and loss curves, a classification report, a real-vs-predicted scatter plot, and a normalised confusion matrix.

## Model Summary

| Property | Value |
| --- | --- |
| Total parameters | 2,612,134 (about 2.61M) |
| Trainable parameters | 354,150 (about 354k) |
| Frozen (backbone) parameters | 2,257,984 |
| Input size | 224 x 224 x 3 |
| Output classes | 102 |

## Key Results

| Metric | Result |
| --- | --- |
| Test accuracy | about 0.91 (819 test images) |
| Macro F1 | 0.90 |
| Weighted F1 | 0.91 |
| Validation accuracy (plateau) | 0.879 to 0.885 |
| Training accuracy (final) | 0.9738 |
| Validation loss (plateau) | about 0.4229 |

The model transfers well early (validation accuracy reaches about 0.86 within three epochs) then saturates, with the gap between training and validation accuracy showing the frozen backbone stops improving its representations. Classes with very few test samples show unstable recall, and some low-contrast or yellow flowers get confused, pointing to a reliance on coarse colour cues where fine texture is needed.

## Repository Structure

```
.
├── A00057300_ML_CW2_code.ipynb   # full pipeline (data, model, training, evaluation)
├── app.py                        # Streamlit demo for single-image prediction
├── requirements.txt              # dependencies
└── README.md                     # this file
```

## How to Run

### Notebook (recommended, Google Colab)

1. Open `A00057300_ML_CW2_code.ipynb` in Google Colab.
2. Set the runtime to a GPU (Runtime > Change runtime type > T4 GPU).
3. Run all cells top to bottom. The notebook downloads the dataset, builds the splits, trains the model, and saves it as `flowers_mobilenetv2.h5`.

### Local

```
pip install -r requirements.txt
jupyter notebook A00057300_ML_CW2_code.ipynb
```

### Streamlit Demo

After the model file `flowers_mobilenetv2.h5` exists:

```
streamlit run app.py
```

Upload a flower image and the app returns the predicted class (0 to 101) with a confidence score.

## Future Work

Selectively unfreeze deeper MobileNetV2 layers, add colour and contrast augmentation, and use class-balanced or focal loss to lift performance on minority classes.

## References

- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR 2018.
- Nilsback and Zisserman, Oxford 102 Flower dataset, VGG, University of Oxford.

## AI Declaration

All modelling, coding, experimentation, and analysis are the author's own work. Generative AI tools were used only for language refinement and to clarify code concepts; the design decisions, interpretations, and conclusions are entirely the author's own.
