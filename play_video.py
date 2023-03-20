import numpy as np
import moviepy
import moviepy.editor as mp
import pygame
import time
from moviepy.audio.AudioClip import AudioArrayClip


if __name__ == '__main__':
    clip = mp.VideoFileClip("generated_video.mp4")
    sound = pygame.mixer.Sound("2.mp3")
    sound.play()
    clip.without_audio().preview()
    while pygame.mixer.get_busy():  # 在音频播放为完成之前不退出程序
        pass
