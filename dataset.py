import os
import cv2
import torchvision
from PIL import Image
from torch.utils.data import Dataset

class OrganoidDataset(Dataset):
    def __init__(self, images_path: str, mask_path: str=None, preprocess_path: str=None, mode: str='train', transform: torchvision.transforms=None):
        self.transform = transform
        self.preprocess_path = preprocess_path
        self.mask_path = mask_path

        self._parser_name(images_path)

        if mode == 'train':
            self.images = [os.path.join(os.path.join(images_path, 'TRAIN'), image) for image in os.listdir(os.path.join(images_path, 'TRAIN')) if image.startswith('IMG')]
            if preprocess_path is not None:
                self.masks = [os.path.join(os.path.join(preprocess_path, 'TRAIN'), image) for image in os.listdir(os.path.join(preprocess_path, 'TRAIN'))]
            elif mask_path is not None:
                self.masks = [os.path.join(os.path.join(mask_path, 'TRAIN'), image) for image in os.listdir(os.path.join(mask_path, 'TRAIN')) if image.startswith('MASK')]
        elif mode == 'val':
            self.images = [os.path.join(os.path.join(images_path, 'VAL'), image) for image in os.listdir(os.path.join(images_path, 'VAL')) if image.startswith('IMG')]
            if preprocess_path is not None:
                self.masks = [os.path.join(os.path.join(preprocess_path, 'VAL'), image) for image in os.listdir(os.path.join(preprocess_path, 'VAL'))]
            elif mask_path is not None:
                self.masks = [os.path.join(os.path.join(mask_path, 'VAL'), image) for image in os.listdir(os.path.join(mask_path, 'VAL')) if image.startswith('MASK')]
        else: 
            self.images = [os.path.join(os.path.join(images_path, 'TEST'), image) for image in os.listdir(os.path.join(images_path, 'TEST')) if image.startswith('IMG')]
            if preprocess_path is not None:
                self.masks = [os.path.join(os.path.join(preprocess_path, 'TEST'), image) for image in os.listdir(os.path.join(preprocess_path, 'TEST'))]
            elif mask_path is not None:
                self.masks = [os.path.join(os.path.join(mask_path, 'TEST'), image) for image in os.listdir(os.path.join(mask_path, 'TEST')) if image.startswith('MASK')]
        

    def __getitem__(self, index):
        img = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        if self.preprocess_path is not None or self.mask_path is not None:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

            img = Image.fromarray(img)
            mask = Image.fromarray(mask)

            if self.transform is not None:
                img = self.transform(img)
                mask = self.transform(mask)

            return img, mask
        else:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            return img

    def __len__(self):
        return len(self.images)
    
    def _parser_name(self, path: str) -> str:
        dirs = os.listdir(path)
        for dir in dirs:
            files = os.listdir(os.path.join(path, dir))
            for file in files:
                if file.startswith('IMG') == False and file.find('img') != -1:
                    os.rename(os.path.join(os.path.join(path, dir), file), os.path.join(os.path.join(path, dir), 'IMG_' + file))
                elif file.startswith('MASK') == False and file.find('masks') != -1:
                    os.rename(os.path.join(os.path.join(path, dir), file), os.path.join(os.path.join(path, dir), 'MASK_' + file))