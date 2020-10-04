import os
import sys
import re

import numpy as np
import skvideo.io
from imageio import imread

if len(sys.argv) < 2:
    print("Pass path to data")
    sys.exit(1)


def get_color_images(images):
    return list(filter(lambda file: bool(re.match("^color_\d{3}\.jpg$", file)), images))


path = sys.argv[1]

cwd = os.getcwd()
os.chdir(path)

speakers = os.listdir(os.getcwd())

for spk in speakers:
    contents = os.listdir(spk + ("/words"))
    for content in contents:
        utterances = os.listdir(os.path.join(spk, "words", content))
        for utterance in utterances:
            with skvideo.io.FFmpegWriter(
                os.path.join(
                    spk,
                    "words",
                    content,
                    utterance,
                    "word%02dutterance%02d.mp4" % (int(content), int(utterance)),
                )
            ) as video:
                print(".", end="")
                images_list = get_color_images(
                    os.listdir(os.path.join(spk, "words", content, utterance))
                )
                n_frames = len(images_list)
                images = np.array(
                    [
                        imread(os.path.join(spk, "words", content, utterance, file))
                        for file in images_list
                    ]
                )
                for img in images:
                    video.writeFrame(img)

os.chdir(cwd)
