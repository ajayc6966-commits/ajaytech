# Emotion Detection Project (FER2013 + CNN)

## Steps
1. Place your Kaggle FER2013 folders inside **data/** like:
   data/
     train/
     test/

2. Install dependencies:
   pip install -r requirements.txt

3. Train model:
   python train.py

4. Run real‑time detector:
   python realtime.py

## Dataset Setup

This project uses the FER2013 emotion dataset.

Due to large file size, the dataset is not uploaded to GitHub.

### Folder Structure
Create the following structure inside the project directory:

data/
├── train/
│   ├── angry
│   ├── happy
│   ├── sad
│   ├── neutral
│   ├── surprise
│   └── fear
└── test/
    ├── angry
    ├── happy
    ├── sad
    ├── neutral
    ├── surprise
    └── fear
