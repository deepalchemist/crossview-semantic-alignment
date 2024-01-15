

# Last Change:  2023-02-04 20:55:05

import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
import torchvision.transforms as tv_trans
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=(224, 224)):
        self.centercrop = centercrop
        if isinstance(size, (tuple, list)):
            assert len(size)==2
            self.size_h, self.size_w = size
        else:
            assert isinstance(size, int)
            self.size_h, self.size_w = size, size
        self.transform = self._transform((self.size_h, self.size_w))

    def _transform(self, n_px):
        return Compose([
            #Resize(n_px, interpolation=Image.BICUBIC),
            Resize(n_px, interpolation=tv_trans.InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, framepath_lst, preprocess):
        images = list()
        for _framepath in framepath_lst:
            try:
                frame = cv2.imread(_framepath, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
            except Exception as e:
                print(f"Bad file path: {_framepath}, error: {e}")

        assert len(images) > 0, f"Bad path: {framepath_lst}"
        video_data = th.tensor(np.stack(images))
        return {'video': video_data}

    def get_video_data(self, video_path):
        image_input = self.video_to_tensor(video_path, self.transform)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2
