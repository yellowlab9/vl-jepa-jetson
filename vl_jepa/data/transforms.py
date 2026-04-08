from torchvision import transforms

def get_train_transforms(config_or_size=224):
    """
    Returns torchvision transforms for the vision encoder during training.
    Accepts either an integer size or a configuration dictionary.
    """
    if isinstance(config_or_size, dict):
        image_size = config_or_size.get('image_size', 224)
    else:
        image_size = config_or_size

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(config_or_size=224):
    """
    Returns torchvision transforms for the vision encoder during validation.
    Accepts either an integer size or a configuration dictionary.
    """
    if isinstance(config_or_size, dict):
        image_size = config_or_size.get('image_size', 224)
    else:
        image_size = config_or_size

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])