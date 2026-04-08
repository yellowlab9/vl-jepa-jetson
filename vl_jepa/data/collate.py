import torch

def jepa_collate_fn(batch):
    """
    Collates a list of dataset dictionaries into a single batch dictionary.
    """
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask  # Changed to singular to match train.py
    }