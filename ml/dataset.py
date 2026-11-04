import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import config

def crop_image_from_gray(img, tol=7):
    """
    Crops out the uninformative black borders around retinal fundus images.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark or empty, return original
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def preprocess_ben_graham(img, img_size=config.IMG_SIZE, sigmaX=30):
    """
    Applies Ben Graham's method (standard competition preprocessing for DR):
    1. Removes empty black borders
    2. Resizes to square image
    3. Blends image with Gaussian blur to normalize color lighting across cameras
    """
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_size, img_size))
    # Local contrast enhancement
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    return enhanced

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, is_train=True, use_ben_graham=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        self.use_ben_graham = use_ben_graham

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocessing
        if self.use_ben_graham:
            img = preprocess_ben_graham(img, img_size=config.IMG_SIZE)
        else:
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

        # Augmentation for training: retinal images are circular & rotational invariant
        if self.is_train:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)  # horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)  # vertical flip
            # Random 90-degree rotations
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, rot_k).copy()

        # Normalize to [0, 1] and convert HWC to CHW tensor
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        if 'diagnosis' in row:
            label = int(row['diagnosis'])
            return img_tensor, torch.tensor(label, dtype=torch.long)
        return img_tensor

def get_dataloaders(csv_path=config.TRAIN_CSV, img_dir=config.TRAIN_IMAGES_DIR, val_split=config.VAL_SPLIT):
    """
    Creates stratified train and validation data loaders preserving the 5-class ratios.
    """
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=config.RANDOM_SEED,
        stratify=df['diagnosis']
    )

    train_ds = APTOSDataset(train_df, img_dir, is_train=True)
    val_ds = APTOSDataset(val_df, img_dir, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_df, val_df
