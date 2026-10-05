# Dataset folder

Place the BUSI dataset here (it is not included in the repository).

1. Download from Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
   (free Kaggle account required).
2. Unzip into this folder so you end up with:

```
ml/dataset/Dataset_BUSI_with_GT/benign/
ml/dataset/Dataset_BUSI_with_GT/malignant/
ml/dataset/Dataset_BUSI_with_GT/normal/
```

Any folder name works as long as it contains `benign`, `malignant` and `normal`.
Then run `python prepare_dataset.py` from the `ml/` folder.
