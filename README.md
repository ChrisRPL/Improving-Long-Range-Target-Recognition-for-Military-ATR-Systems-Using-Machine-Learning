# Improving-Long-Range-Target-Recognition-for-Military-ATR-Systems-Using-Machine-Learning
Code implementation of paper: Improving Long-Range Target Recognition for Military ATR Systems Using Machine Learning

## Project Structure

```plaintext
├── LICENSE
├── models
│   ├── __init__.py
│   ├── dsgan.py                # DS-GAN model for downsampling
│   ├── detection_model.py      # Enhanced detection model
├── README.md
├── scripts
│   ├── __init__.py
│   ├── prepare_dsgan_dataset.py  # Script to prepare dataset for DS-GAN training
│   ├── train_dsgan.py            # Script to train DS-GAN
│   ├── inpaint_with_gan.py       # Script to inpaint new objects using DS-GAN
│   ├── prepare_fusion_dataset.py # Script to prepare dataset for video fusion
│   ├── prepare_detection_dataset.py  # Script to prepare dataset for detection model training
│   ├── train_detection_model.py  # Script to train the enhanced detection model
├── utils
│   ├── __init__.py
│   ├── video_processing.py
│   ├── annotations.py            # Annotation utilities
├── data
│   ├── mwir_videos/
│   ├── visible_videos/
│   ├── fused_videos/
│   ├── dsgan_videos/
│   ├── annotations/
│   ├── gan_dataset/
│       ├── high_res/
│       ├── low_res/
│   ├── detection_dataset/
│       ├── images/
│       ├── flows/
│       ├── annotations/
└── outputs/
```

## Setup

### Install the required packages:

```sh
pip install -r requirements.txt
```

## Preparing Datasets

### 1. DS-GAN Dataset Preparation

To prepare the dataset for DS-GAN training:

```sh
python scripts/prepare_dsgan_dataset.py --video_path data/rgb_videos/sample_video.avi --annotation_path data/annotations/sample_annotations.txt --high_res_output_dir data/gan_dataset/high_res --low_res_output_dir data/gan_dataset/low_res
```

### 2. DS-GAN Training

To train the DS-GAN model:

```sh
python scripts/train_dsgan.py --high_res_dir data/gan_dataset/high_res --low_res_dir data/gan_dataset/low_res
```

### 3. Inpainting with GAN

To inpaint new objects using the trained DS-GAN and update the annotations:

```sh
python scripts/inpaint_with_gan.py --video_path data/rgb_videos/sample_video.avi --annotation_path data/annotations/sample_annotations.txt --output_video_path data/inpainted_video.avi --output_annotation_path data/inpainted_annotations.txt --model_path models/dsgan_generator_200.pth
```

### 4. Fusion Dataset Preparation

To create the dataset for video fusion:

```sh
python scripts/prepare_fusion_dataset.py --visible_video_path data/visible_videos/sample_visible_video.avi --mwir_video_path data/mwir_videos/sample_mwir_video.avi --output_video_path data/fused_videos/sample_fused_video.avi
```

### 5. Detection Dataset Preparation

To create the dataset for detection model training:

```sh
python scripts/prepare_detection_dataset.py --video_path data/rgb_videos/sample_video.avi --annotation_path data/annotations/sample_annotations.txt --output_image_dir data/detection_dataset/images --output_flow_dir data/detection_dataset/flows --output_annotation_dir data/detection_dataset/annotations
```

## Training the Detection Model

To train the enhanced detection model on the chosen dataset:

```sh
python scripts/train_detection_model.py --dataset_type fused --num_classes 91
```

## Directory Overview

- **models/**: Contains the model definitions for DS-GAN and the enhanced detection model.
- **scripts/**: Contains scripts for preparing datasets, training models, and inpainting using DS-GAN.
- **utils/**: Contains utility functions for video processing and handling annotations.
- **data/**: Contains directories for storing videos, annotations, and prepared datasets.
- **outputs/**: Directory to save trained models and results.

## Contributing

If you would like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
