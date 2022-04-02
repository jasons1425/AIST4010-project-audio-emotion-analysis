from torchvision import transforms
import numpy as np


class FreqMasking(object):
    def __init__(self, prob=0.5, portion=0.1):
        self.prob = prob
        self.portion = portion

    def __call__(self, sample):
        if np.random.uniform() > self.prob:
            return sample
        # expect the spectrogram to be in shape of (RGB channel * frequency * time)
        sample = sample.permute(1, 2, 0)  # frequency * time * RGB
        freq_range, time_range = sample.shape[:2]
        mask_area = int((freq_range * self.portion) // 1)
        mask_start = np.random.randint(mask_area, freq_range-mask_area)
        sample[mask_start: mask_start+mask_area] = 0
        sample = sample.permute(2, 0, 1)  # RGB * frequency * time
        return sample


class TimeMasking(object):
    def __init__(self, prob=0.5, portion=0.1):
        self.prob = prob
        self.portion = portion

    def __call__(self, sample):
        if np.random.uniform() > self.prob:
            return sample
        # expect the spectrogram to be in shape of (RGB channel * frequency * time)
        sample = sample.permute(2, 1, 0)  # time * frequency * RGB
        time_range, freq_range = sample.shape[:2]
        mask_area = int((time_range * self.portion) // 1)
        mask_start = np.random.randint(mask_area, time_range-mask_area)
        sample[mask_start: mask_start+mask_area] = 0
        sample = sample.permute(2, 1, 0)  # RGB * frequency * time
        return sample


def spectrum_transform(resize=224, crop_size=None, norm=True, freq_mask=(0.5, 0.1), time_mask=(0.5, 0.1)):
    trans_stack = [transforms.ToTensor()]
    if crop_size:
        trans_stack.append(transforms.RandomCrop(crop_size))
    if resize:
        trans_stack.append(transforms.Resize(resize))
    if freq_mask:
        trans_stack.append(FreqMasking(*freq_mask))
    if time_mask:
        trans_stack.append(TimeMasking(*time_mask))
    if norm:
        trans_stack.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    trans = transforms.Compose(trans_stack)
    return trans
