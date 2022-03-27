from torchvision import transforms


def spectrum_transform():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trans


def spectrum_transform_crop(crop_size=200):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(crop_size),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trans
