import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)


class WaveGANGenerator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1,
                 latent_dim=128, post_proc_filt_len=512,
                 verbose=False, upsample=True):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.latent_di = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose
        # "Dense" is the same meaning as fully connection.
        self.fc1 = nn.Linear(latent_dim, 256 * model_size)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4
        self.deconv_1 = Transpose1dLayer(16 * model_size, 8 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(8 * model_size, 4 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(4 * model_size, 2 * model_size, 25, stride, upsample=upsample)
        self.deconv_4 = Transpose1dLayer(2 * model_size, model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Transpose1dLayer(model_size, num_channels, 25, stride, upsample=upsample)

        if post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_1(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_2(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_3(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.deconv_4(x))
        if self.verbose:
            print(x.shape)

        output = F.tanh(self.deconv_5(x))
        return output


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass


class WaveGANDiscriminator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, shift_factor=2,
                 alpha=0.2, verbose=False):
        super(WaveGANDiscriminator, self).__init__()
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.fc1 = nn.Linear(256 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        x = x.view(-1, 256 * self.model_size)
        if self.verbose:
            print(x.shape)

        return self.fc1(x)



class PitchClassifier(nn.Module):
    def __init__(self):
        super(PitchClassifier, self).__init__()
        self.verbose = False

        self.conv1 = nn.Conv2d(2, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, [3,3], padding=1)
        self.conv3 = nn.Conv2d(32, 32, [3,3], padding=1)
        #Downsample
        self.down1 = nn.Conv2d(32, 32, [2,2], stride=2)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        #Downsample
        self.down2 = nn.Conv2d(64, 64, 2, stride=2)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        #Downsample
        self.down3 = nn.Conv2d(128, 128, 2, stride=2)
        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        #Downsample
        self.down4 = nn.Conv2d(256, 256, 2, stride=2)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        #Downsample
        self.down5 = nn.Conv2d(256, 256, 2, stride=2)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv13 = nn.Conv2d(256, 256, 3, padding=1)
        #Downsample
        self.down6 = nn.Conv2d(256, 256, 2, stride=2)
        #concat(x, minibatch std.)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv15 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(2*16*256, 61)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        #conv1
        x = F.leaky_relu(self.conv1(x))
        if self.verbose:
            print("conv1", x.shape[2], x.shape[3], x.shape[1])

        #conv2
        x = F.leaky_relu(self.conv2(x))
        if self.verbose:
            print("conv2", x.shape[2], x.shape[3], x.shape[1])

        #conv3
        x = F.leaky_relu(self.conv3(x))
        if self.verbose:
            print("conv3", x.shape[2], x.shape[3], x.shape[1])

        #down1
        x = self.down1(x)
        if self.verbose:
            print("down1", x.shape[2], x.shape[3], x.shape[1])

        #conv4
        x = F.leaky_relu(self.conv4(x))
        if self.verbose:
            print("conv4", x.shape[2], x.shape[3], x.shape[1])

        #conv5
        x = F.leaky_relu(self.conv5(x))
        if self.verbose:
            print("conv5", x.shape[2], x.shape[3], x.shape[1])

        #down2
        x = self.down2(x)
        if self.verbose:
            print("down2", x.shape[2], x.shape[3], x.shape[1])

        #conv6
        x = F.leaky_relu(self.conv6(x))
        if self.verbose:
            print("conv6", x.shape[2], x.shape[3], x.shape[1])

        #conv7
        x = F.leaky_relu(self.conv7(x))
        if self.verbose:
            print("conv7", x.shape[2], x.shape[3], x.shape[1])

        #down3
        x = self.down3(x)
        if self.verbose:
            print("down3", x.shape[2], x.shape[3], x.shape[1])

        #conv8
        x = F.leaky_relu(self.conv8(x))
        if self.verbose:
            print("conv8", x.shape[2], x.shape[3], x.shape[1])

        #conv9
        x = F.leaky_relu(self.conv9(x))
        if self.verbose:
            print("conv9", x.shape[2], x.shape[3], x.shape[1])

        #down4
        x = self.down4(x)
        if self.verbose:
            print("down4", x.shape[2], x.shape[3], x.shape[1])

        #conv10
        x = F.leaky_relu(self.conv10(x))
        if self.verbose:
            print("conv10", x.shape[2], x.shape[3], x.shape[1])


        #conv11
        x = F.leaky_relu(self.conv11(x))
        if self.verbose:
            print("conv11", x.shape[2], x.shape[3], x.shape[1])

        #down5
        x = self.down5(x)
        if self.verbose:
            print("down5", x.shape[2], x.shape[3], x.shape[1])

        #conv12
        x = F.leaky_relu(self.conv12(x))
        if self.verbose:
            print("conv12", x.shape[2], x.shape[3], x.shape[1])

        #conv13
        x = F.leaky_relu(self.conv13(x))
        if self.verbose:
            print(x.shape)
            print("conv13", x.shape[2], x.shape[3], x.shape[1])

        #down6
        x = self.down6(x)
        if self.verbose:
            print("down6", x.shape[2], x.shape[3], x.shape[1])

        #conv14
        x = F.leaky_relu(self.conv14(x))
        if self.verbose:
            print("conv14", x.shape[2], x.shape[3], x.shape[1])

        #conv15
        x = F.leaky_relu(self.conv15(x))
        if self.verbose:
            print("conv15", x.shape[2], x.shape[3], x.shape[1])

        x = x.view(2,-1)
        x = F.softmax(self.fc1(x))
        if self.verbose:
            print("output", x.shape)

        return x