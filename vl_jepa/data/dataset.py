import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from .transforms import get_train_transforms, get_val_transforms

class COCOCaptionsDataset(Dataset):
    def __init__(self, data_dir, split='train2017', transform=None, max_samples=None):
        self.image_dir = os.path.join(data_dir, 'images', split)
        ann_file = os.path.join(data_dir, 'annotations', f'captions_{split}.json')
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
            
        # Map image IDs to their filenames for quick lookup
        self.images = {img['id']: img['file_name'] for img in coco_data['images']}
        self.annotations = coco_data['annotations']
        
        # Support dataset subsetting for testing
        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]
            
        self.transform = transform
        
        # DistilBERT tokenizer config (max length 128 as specified in the model architecture)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 128

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_id = ann['image_id']
        caption = ann['caption']
        
        # Load and transform image
        img_path = os.path.join(self.image_dir, self.images[img_id])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

def create_dataset(data_dir, split='train2017', is_train=True, max_samples=None, transform=None, **kwargs):
    """
    Helper function to initialize the dataset.
    Accepts an explicit transform if train.py provides one.
    Handles cases where train.py passes a config dictionary instead of a string path.
    """
    # Extract properties if train.py passed a dictionary
    if isinstance(data_dir, dict):
        config = data_dir
        
        # Extract the actual data_root path
        actual_data_dir = config.get('data_root', './data')
        
        # The config uses 'train' or 'val', but COCO folders are 'train2017'/'val2017'
        config_split = config.get('train_split' if is_train else 'val_split', split)
        if config_split in ['train', 'val']:
            split = f"{config_split}2017"
        else:
            split = config_split
            
        # Get max_samples from config if not explicitly provided
        if max_samples is None:
            max_samples = config.get('max_samples', None)
            
        # Reassign data_dir to the string path
        data_dir = actual_data_dir

    # If train.py didn't pass a transform, build it here
    if transform is None:
        if is_train:
            transform = get_train_transforms()
        else:
            transform = get_val_transforms()
            
    return COCOCaptionsDataset(data_dir, split, transform, max_samples)