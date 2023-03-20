import os
import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch
import dataset
from dataset import AutoVoDataset
from models.vae import VQVAE
from models.autovo import AutoVo
from torch.utils import data
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_save_path = "./params/autovo.pth"
loader = data.DataLoader(dataset=AutoVoDataset(), batch_size=1, shuffle=False, num_workers=0)
print(len(loader.dataset))

vae_params_path = "./params/vae.pth"
vae = VQVAE(in_dim=3, num_embeddings=512, embedding_dim=256)
vae.to(device)
vae.load_state_dict(torch.load(vae_params_path))
vae.eval()

model = AutoVo()
model.to(device)
model.load_state_dict(torch.load(params_save_path))
model.eval()


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def generate_frames():
    def tensor_to_img(x):
        b, h, w, c = x.shape[0], x.shape[2], x.shape[3], x.shape[1]
        x = (x.permute([0, 2, 3, 1]).contiguous().view(-1, 3) * torch.Tensor(dataset.std).to(device).view(1, 3) + torch.Tensor(dataset.mean).to(device).view(1, 3)) * 255
        x = x.contiguous().view(b, h, w, c)
        x = x.cpu().numpy().astype(np.uint8)
        return [cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB) for i in range(x.shape[0])]
    with torch.no_grad():
        processed = 0
        for audio, imgs in loader:
            audio, imgs = audio.to(device), imgs.to(device)
            recon = model(audio, imgs, vae)
            recon = tensor_to_img(recon.squeeze(0))
            for i in range(len(recon)):
                processed += 1
                cv2.imwrite("./frames/%d.jpg" % processed, recon[i])
                print("\r进度：%s" % (processbar(processed, len(loader.dataset)*dataset.video.fps)), end="")
    print()


def merge_frames():
    fps = 30
    image_files = ["./frames/%d.jpg" % (i+1) for i in range(len(loader.dataset)*int(dataset.video.fps))]
    generated_video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    generated_video.write_videofile("generated_video.mp4", fps=fps)


if __name__ == '__main__':
    # generate_frames()
    merge_frames()

    # import moviepy.editor as mp
    # video = mp.VideoFileClip("generated_video.mp4")
    # video = video.set_audio(dataset.video.audio.subclip(0, int(video.duration)))
    # video.write_videofile("generated_video_with_audio.mp4", fps=30)