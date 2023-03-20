import numpy as np
import moviepy.editor as mp
from torch.utils import data
import torch
import cv2
from PIL import Image
from torchvision import transforms

# 480, 852
video = mp.VideoFileClip("1.mp4")
# mean = [0.22060803, 0.19703749, 0.17213199]
# std = [0.29451922, 0.27345984, 0.2690363]
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


def compute_mean_and_std():
    frames = [cv2.resize(frame, dsize=(213, 120)).astype(np.float32).reshape(-1, 3)/255 for frame in video.iter_frames()]
    n = len(frames)*frames[0].shape[0]
    mean = np.zeros(shape=(3, ))
    for i in range(len(frames)):
        mean += np.sum(frames[i], axis=0)
    mean /= n
    std = np.zeros(shape=(3, ))
    for i in range(len(frames)):
        std += np.sum((frames[i]-mean)**2, axis=0)
    std = np.sqrt(std / n)
    print(mean, std)
    # [0.22060803 0.19703749 0.17213199] [0.29451922 0.27345984 0.2690363 ]
    return mean, std


class VAEDataset(data.Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.frames = [self.transform(Image.fromarray(cv2.resize(frame, dsize=(216, 120)))) for frame in video.iter_frames()]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class AutoVoDataset(data.Dataset):
    def __init__(self, sep_seconds=1):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.sep_seconds = sep_seconds
        self.length = int(video.duration)-self.sep_seconds*10+1
        self.mean = np.mean(video.audio.to_soundarray(), axis=0, keepdims=True)
        self.std = np.std(video.audio.to_soundarray(), axis=0, keepdims=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frames = [self.transform(Image.fromarray(cv2.resize(frame, dsize=(216, 120)))) for frame in video.subclip(index, index+self.sep_seconds).iter_frames()]
        # audio_len x 2
        audio = video.audio.subclip(index, index+self.sep_seconds*10).to_soundarray()
        audio = (audio - self.mean) / self.std
        audio = torch.Tensor(audio.astype(np.float32))
        # L x 3 x h x w
        frames = torch.stack(frames, dim=0)
        return audio, frames


if __name__ == '__main__':
    vae_dataset = VAEDataset()
    print(len(vae_dataset))