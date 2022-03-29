from torchvision import transforms


def spectrum_transform(resize=224, crop_size=None, norm=True):
    trans_stack = [transforms.ToTensor()]
    if crop_size:
        trans_stack.append(transforms.RandomCrop(crop_size))
    if resize:
        trans_stack.append(transforms.Resize(resize))
    if norm:
        trans_stack.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    trans = transforms.Compose(trans_stack)
    return trans
