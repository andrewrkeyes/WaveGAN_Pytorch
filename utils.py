import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
import re
# import torchaudio as ta
import pickle
import lmdb
import os.path as osp
from scipy.signal import resample
# from moviepy.editor import VideoFileClip
from torchvision import transforms, io
import librosa as ta
from librosa import core as ap
from functools import partial
import librosa
import pandas as pd
import spectral_ops
import tensorflow as tf

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    return dl


class AudioDirectoryDataset(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = Path(self.data_dir).glob('**/*.wav')
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = str(file_name)
            if os.path.isfile(possible_file) and '.wav' in possible_file:
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, csv_file, transform=None, fps=16000, tf_sess=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform
        self.fps = fps
        self.resample = partial(librosa.core.resample, 16000, fps)
        csv_file = os.path.join(data_dir, csv_file)
        self.mem_frame = pd.read_csv(csv_file)
        self.tf_sess = tf_sess
        self.spectral_params = Dict(
            waveform_length=self.fps*4,
            sample_rate=self.fps,
            spectrogram_shape=[224, 1024],
            overlap=0.75
        )

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.mem_frame)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        # read the image:
        #audio = self.files[idx]
        audio = os.path.join(self.data_dir, self.mem_frame.iloc[idx, 0])
        target = self.mem_frame.iloc[idx, 1] - 24
        audio, sr = librosa.core.load(audio)
        audio = librosa.core.resample(audio, sr, self.fps)
        audio = np.reshape(audio, (1,-1))
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=1)
        if audio.shape[-1] < self.fps:
            n_samples = self.fps
            total_samples = audio.shape[-1]
            n_pad = n_samples - total_samples
            audio = torch.nn.functional.pad(audio.view(1, 1, -1), (0, n_pad), mode='replicate').view(1, -1)
        elif audio.shape[-1] > self.fps:
            audio = audio[:, 0:self.fps]

        audio, audio_it = spectral_ops.convert_to_spectrogram(audio, **self.spectral_params)
        audio = tf.stack([audio, audio_it], axis=1)
        audio = tf.reshape(audio, [2, 128, 1024])
        audio = torch.Tensor(self.tf_sess.run(audio))
        return audio, target


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class VideoDirectoryDataset(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory dataset """

    def __init__(self, data_dir, audio_fps=16384, image_size=224, video_fps=16):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.audio_fps = audio_fps
        self.image_size = image_size
        self.video_fps = video_fps

        self.classes = os.listdir(data_dir)
        self.classes.sort()

        file_names = Path(self.data_dir).glob('**/*.mp4')
        self.files = []  # initialize to empty list
        for file_name in file_names:
            self.files.append(str(file_name))
        print(f"We have {len(self.files)} videos in total!")

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def get_class(self, x):
        for i in self.classes:
            if i in x:
                return self.classes.index(i)

    def preprocess_video(self, x):
        if x.shape[0] > self.video_fps:
            x = x[np.sort(np.random.choice(list(range(x.shape[0])), self.video_fps, False))]
        elif x.shape[0] < self.video_fps:
            x = x[np.sort(np.random.choice(list(range(x.shape[0])), self.video_fps))]
        x = torch.nn.functional.interpolate(x.float().permute(0, 3, 1, 2),
                                            (self.image_size, self.image_size), mode='bilinear', align_corners=True)
        if self.transform:
            x = self.transform(x)
        return x

    def preprocess_audio(self, x, audio_fps):
        x = ap.to_mono(x.numpy())
        if audio_fps != self.audio_fps:
            x = ap.resample(x, audio_fps, self.audio_fps)
        if x.shape[0] < self.audio_fps:
            x = np.pad(x, (0, self.audio_fps - x.shape[0]))
        return x.reshape((1, -1))

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        rand = np.random.RandomState()
        video = self.files[idx]
        target = self.get_class(video)
        video, audio, info = io.read_video(video)

        video_duration = video.shape[0] // info['video_fps']

        # Less than one second video
        if video_duration == 0:
            return self.preprocess_video(video), self.preprocess_audio(audio, info['audio_fps']), target
        else:
            second_idx = rand.randint(0, video_duration - 1)
            second_idx_vid = int(second_idx * info['video_fps'])
            second_idx_aud = int(second_idx * info['audio_fps'])

            video = video[second_idx_vid:second_idx_vid + int(info['video_fps'])]
            audio = audio[:, second_idx_aud:second_idx_aud + int(info['audio_fps'])]

            return self.preprocess_video(video), self.preprocess_audio(audio, info['audio_fps']), target


class VideoDirectoryDataset2(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory dataset """

    def __init__(self, data_dir, audio_fps=16384, image_size=224, video_fps=16):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transforms.Compose([ToTensor(),
                                             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.audio_fps = audio_fps
        self.image_size = image_size
        self.video_fps = video_fps

        self.classes = os.listdir(data_dir)
        self.classes.sort()

        file_names = Path(self.data_dir).glob('**/*.mp4')
        self.files = []  # initialize to empty list
        for file_name in file_names:
            self.files.append(str(file_name))
        print(f"We have {len(self.files)} videos in total!")

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def get_class(self, x):
        for i in self.classes:
            if i in x:
                return self.classes.index(i)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        rand = np.random.RandomState()

        try:
            video = self.files[idx]
            target = self.get_class(video)
            video = VideoFileClip(video, audio_fps=self.audio_fps, target_resolution=(self.image_size, self.image_size))
            video_fps = int(video.fps // 1)
            video_duration = video.duration // 1
            audio = video.audio.to_soundarray()
            video = np.array(list(video.iter_frames()))

            second_idx = rand.randint(0, video_duration - 1)
            second_idx_vid = second_idx * video_fps
            second_idx_aud = second_idx * self.audio_fps

            video = video[second_idx_vid:second_idx_vid + video_fps]
            video = resample(video, self.video_fps, axis=0)
            if self.transform:
                video = self.transform(torch.tensor(video).permute(0, 3, 1, 2))
            audio = audio[second_idx_aud:second_idx_aud + self.audio_fps]
            if audio.shape[1] > 1:
                audio = audio.mean(1).reshape((1, -1))
            return video, audio, target
        except:
            print(self.files[idx])
            return self.__getitem__(4)


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        return pickle.loads(byteflow)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class Dict(dict):

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]
