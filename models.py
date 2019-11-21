# Import necessary modules
import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

# Constraints
# Input: [batch_size, in_channels, height, width]
class SpectralReg(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralReg, self).__init__()
        self.module = module
        self.name = name
        # self.power_iterations = power_iterations
        # if not self._made_params():
        #     self._make_params()

    def _update_u_v(self):
        # u = getattr(self.module, self.name + "_u")
        # v = getattr(self.module, self.name + "_v")
        # w = getattr(self.module, self.name + "_bar")
        # try:
        w = getattr(self.module, self.name + "_orig")
        # print(type(w), "-------")
        shape = w.shape

        height = w.data.shape[0]
        w_mat = w.data.reshape(height, -1)
        device = w_mat.device

        # svd with different runtimes

        # U, s, V = torch.svd(w_mat.data.cpu())
        # U, s, V = np.linalg.svd(w_mat.data.cpu())
        # U = torch.from_numpy(s).to(device)
        # s = torch.from_numpy(s).to(device)
        # V = torch.from_numpy(s).to(device)
        U, s, V = np.linalg.svd(w_mat.cpu(), full_matrices=False)

        # U = torch.from_numpy(U).to(device)
        # s = torch.from_numpy(s).to(device)
        # V = torch.from_numpy(V).to(device)



        sigma1 = max(s)
        s = s / sigma1
        s[:s.shape[0] // 2] = 1
        S = np.diag(s)

        # S = torch.diag(s)
        compensated_w = torch.from_numpy(np.dot(U, np.dot(S, V))).to(device)

        # U, s, V = np.linalg.svd(w_mat.data.detach().cpu(), full_matrices=False)
        #
        # if len(s.shape) != 1:
        #     raise Exception("s has more than 1 dim")
        #
        # sigma1 = max(s)
        # s = s / sigma1
        # s[:s.shape[0] // 2] = 1
        #
        # S = np.diag(s)
        #
        # compensated_w = np.dot(U, np.dot(S, V))
        # compensated_w = torch.mm(U, torch.mm(S, V.transpose(0, 1)))

        w.data = compensated_w.reshape(shape)

        setattr(self.module, self.name, w)
        # except:
        #     print("=================================")
    # def _made_params(self):
    #     try:
    #         # u = getattr(self.module, self.name + "_u")
    #         # v = getattr(self.module, self.name + "_v")
    #         w = getattr(self.module, self.name)
    #         return True
    #     except AttributeError:
    #         return False


    # def _make_params(self):
        # w = getattr(self.module, self.name + "_orig")
        #
        # height = w.data.shape[0]
        # width = w.view(height, -1).data.shape[1]
        #
        # u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        # v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        # u.data = l2normalize(u.data)
        # v.data = l2normalize(v.data)
        # w_bar = Parameter(w.data)
        #
        # del self.module._parameters[self.name + "_orig"]
        #
        # self.module.register_parameter(self.name + "_u", u)
        # self.module.register_parameter(self.name + "_v", v)

        # w = getattr(self.module, self.name + "_orig")

        # del self.module._parameters[self.name + "_orig"]

        # self.module.register_parameter(self.name, w)


    def forward(self, *args):

        self._update_u_v()
        return self.module.forward(*args)



class AESConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(*args, **kwargs)
        self.conv.weight.data.normal_()
        self.conv.bias.data.normal_()

    def forward(self, x):
        return self.conv(x)


class AESDeconv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(*args, **kwargs)
        self.conv.weight.data.normal_()
        self.conv.bias.data.normal_()


    def forward(self, x):
        result = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                                  align_corners=False)
        return self.conv(result)


class AutoEncoder(nn.Module):
    def __init__(self, step):
        super().__init__()

        self.relu = nn.ReLU()
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])

        # Encoder
        # (800, 256, 1) -> (400, 128, 16) -> (200, 64, 32) -> (100, 32, 64) -> (50, 16, 128) -> (25, 8, 256) ->
        self.encoder_layers.append(
            nn.Sequential(AESConv2d(in_channels=1,
                                      out_channels=8,
                                      kernel_size=3,
                                      padding=1,
                                      stride=(2, 2)),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )


        for i in range(step):
            self.encoder_layers.append(nn.Sequential(
                AESConv2d(in_channels=8 * (2 ** i),
                          out_channels=8*(2**(i+1)),
                          kernel_size=3,
                          padding=1,
                          stride=(2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=8*(2**(i+1)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ))

        # self.encoder_layers.append(nn.Sequential(
        #     # (50, 16, 64) -> (25, 8, 128)
        #     AESConv2d(in_channels=64,
        #               out_channels=128,
        #               kernel_size=3,
        #               padding=(1, 1),
        #               stride=(2, 2)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
        #                    track_running_stats=True)
        # ))
        self.encoder_layers.append(nn.Sequential(
            # (25, 8, 128) -> (12, 4, 256)
            AESConv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=(0, 1),
                      stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        ))
        self.encoder_layers.append(nn.Sequential(
            # (12, 4, 256) -> (4, 2, 512)
            AESConv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(5, 3),
                      padding=(2, 1),
                      stride=(3, 2)),

        ))
        self.encoder_layers.append(nn.Sequential(
            # (4, 2, 512) -> (2, 1, 1024)
            AESConv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      padding=(1, 1),
                      stride=(2, 2)),

        ))
        self.encoder_layers.append(nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        ))


        # Decoder
        self.decoder_layers.append(nn.Sequential(
            #  (2, 1, 1024) -> (4, 2, 512)
            nn.ConvTranspose2d(in_channels=1024,
                                out_channels=512,
                                kernel_size=(4, 4),
                                padding=1,
                                stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
        ))
        self.decoder_layers.append(nn.Sequential(

            # (4, 2, 512) -> (12, 4, 256)
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=(5, 4),
                               padding=1,
                               stride=(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
        ))
        self.decoder_layers.append(nn.Sequential(

            # (12, 4, 256) -> (25, 8, 128)
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=(5, 4),
                               padding=1,
                               stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
        ))



        for i in range(step):
            self.decoder_layers.append(nn.Sequential(
                AESDeconv2d(in_channels=8 * (2 ** (step - i)),
                            out_channels=8 * (2 ** (step - i - 1)),
                            kernel_size=3,
                            padding=1,
                            stride=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=8 * (2 ** (step - i - 1)), eps=1e-05, momentum=0.1, affine=True,
                               track_running_stats=True))
            )
        self.decoder_layers.append(nn.Sequential(
                AESDeconv2d(in_channels=8,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=(1, 1)),
        ))


    def forward(self, *input):
        result = input[0]
        result = self.encode(result)

        result = self.relu(result)

        result = self.decode(result)

        return result

    def encode(self, result):

        for i in range(len(self.encoder_layers)):

            result = self.encoder_layers[i](result)
            # print(i, result.shape)



        return result

    def decode(self, result):
        for i in range(len(self.decoder_layers) - 1):
            result = self.decoder_layers[i](result)
            # print(i, result.shape)

        result = self.decoder_layers[-1](result)

        return result


# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


# Uniformly set the hyperparameters of Linears
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out, sr=False):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)
        if sr:
            self.linear = SpectralReg(self.linear)


    def forward(self, x):
        return self.linear(x)


# Uniformly set the hyperparameters of Conv2d
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sr=False):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        conv.weight.data.normal_()
        conv.bias.data.zero_()


        self.conv = quick_scale(conv)

        if sr:
            self.conv = SpectralReg(self.conv)

    def forward(self, x):
        return self.conv(x)


# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


# "learned affine transform" A
class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


# "learned per-channel scaling factors" B
# 5/13: Debug - tensor -> nn.Parameter
class Scale_B(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        result = noise * self.weight
        return result

    # Early convolutional block


# 5/13: Debug - tensor -> nn.Parameter
# 5/13: Remove noise generating module
class Early_StyleConv_Block(nn.Module):
    '''
    This is the very first block of generator that get the constant value as input
    '''

    def __init__(self, n_channel, dim_latent, dim_input):
        super().__init__()
        # Constant input
        self.constant = nn.Parameter(torch.randn(1, n_channel, dim_input[0], dim_input[1]))
        # Style generators
        self.style1 = FC_A(dim_latent, n_channel)
        self.style2 = FC_A(dim_latent, n_channel)
        # Noise processing modules
        self.noise1 = quick_scale(Scale_B(n_channel))
        self.noise2 = quick_scale(Scale_B(n_channel))
        # AdaIn
        self.adain = AdaIn(n_channel)
        self.lrelu = nn.LeakyReLU(0.2)
        # Convolutional layer
        self.conv = SConv2d(n_channel, n_channel, 3, padding=1)


    def forward(self, latent_w, noise):
        # Gaussian Noise: Proxyed by generator
        # noise1 = torch.normal(mean=0,std=torch.ones(self.constant.shape)).cuda()
        # noise2 = torch.normal(mean=0,std=torch.ones(self.constant.shape)).cuda()
        result = self.constant.repeat(noise.shape[0], 1, 1, 1)
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)

        return result


# General convolutional blocks
# 5/13: Remove upsampling
# 5/13: Remove noise generating
class StyleConv_Block(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel, dim_latent, sr=False):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        self.style2 = FC_A(dim_latent, out_channel)
        # Noise processing modules
        self.noise1 = quick_scale(Scale_B(out_channel))
        self.noise2 = quick_scale(Scale_B(out_channel))
        # AdaIn
        self.adain = AdaIn(out_channel)
        self.lrelu = nn.LeakyReLU(0.2)
        # Convolutional layers
        self.conv1 = SConv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = SConv2d(out_channel, out_channel, 3, padding=1)


    def forward(self, previous_result, latent_w, noise):
        # Upsample: Proxyed by generator
        # result = nn.functional.interpolate(previous_result, scale_factor=2, mode='bilinear',
        #                                           align_corners=False)
        # Conv 3*3
        result = self.conv1(previous_result)
        # Gaussian Noise: Proxyed by generator
        # noise1 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        # noise2 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        # Conv & Norm
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv2(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)

        return result

    # Very First Convolutional Block


# 5/13: No more downsample, this block is the same sa general ones
# class Early_ConvBlock(nn.Module):
#     '''
#     Used to construct progressive discriminator
#     '''
#     def __init__(self, in_channel, out_channel, size_kernel, padding):
#         super().__init__()
#         self.conv = nn.Sequential(
#             SConv2d(in_channel, out_channel, size_kernel, padding=padding),
#             nn.LeakyReLU(0.2),
#             SConv2d(out_channel, out_channel, size_kernel, padding=padding),
#             nn.LeakyReLU(0.2)
#         )

#     def forward(self, image):
#         result = self.conv(image)
#         return result

# General Convolutional Block
# 5/13: Downsample is now removed from block module
class ConvBlock(nn.Module):
    '''
    Used to construct progressive discriminator
    '''

    def __init__(self, in_channel, out_channel, size_kernel1, padding1,
                 size_kernel2=None, padding2=None, stride=(1, 1), sr=False, sr_last=False):
        super().__init__()

        if size_kernel2 == None:
            size_kernel2 = size_kernel1
        if padding2 == None:
            padding2 = padding1

        self.conv = nn.Sequential(
            SConv2d(in_channel, out_channel, size_kernel1, padding=padding1, stride=stride, sr=sr),
            nn.LeakyReLU(0.2),
            SConv2d(out_channel, out_channel, size_kernel2, padding=padding2, sr=sr_last),
            nn.LeakyReLU(0.2)
        )


    def forward(self, image):
        # Conv
        result = self.conv(image)
        return result


# Main components
class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(SLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w


class Encoder(nn.Module):
    '''
    Main Module
    '''
    def __init__(self):
        super().__init__()
        # Waiting to adjust the size
        self.from_rgbs = nn.ModuleList([
            SConv2d(1, 16, 1),
            SConv2d(1, 32, 1),
            SConv2d(1, 64, 1),
            SConv2d(1, 128, 1),
            SConv2d(1, 256, 1),
            SConv2d(1, 512, 1),
            SConv2d(1, 512, 1),
            SConv2d(1, 512, 1),
            SConv2d(1, 512, 1)
        ])
        self.convs = nn.ModuleList([
            ConvBlock(16, 32, 3, 1, stride=(2, 2)),
            ConvBlock(32, 64, 3, 1, stride=(2, 2)),
            ConvBlock(64, 128, 3, 1, stride=(2, 2)),
            ConvBlock(128, 256, 3, 1, stride=(2, 2)),
            ConvBlock(256, 512, 3, 1, stride=(2, 2)),
            ConvBlock(512, 512, 3, 1, stride=(2, 2)),
            ConvBlock(512, 512, 3, 1, stride=(2, 2)),
            ConvBlock(512, 512, 3, 1, stride=(2, 2)),
            ConvBlock(512, 512, 3, 1, (25, 8), 0)
        ])
        self.fc1 = SLinear(512, 512)

        self.n_layer = 9  # 9 layers network

    def forward(self, image,
                step=0,  # Step means how many layers (count from 4 x 4) are used to train
                alpha=-1):  # Alpha is the parameter of smooth conversion of resolution):


        if step == 0:
            result = self.from_rgbs[self.n_layer - 1](image)
            result = self.convs[self.n_layer - 1](result)

        else:
            # from RGB of current step
            result = self.from_rgbs[self.n_layer - 1 - step](image)
            result = self.convs[self.n_layer - 1 - step](result)


            half_res_im = nn.functional.interpolate(image, scale_factor=0.5,
                                                    mode='bilinear', align_corners=False)
            result_prev = self.from_rgbs[self.n_layer - step](half_res_im)

            result = alpha * result + (1 - alpha) * result_prev

            for i in range(self.n_layer - step, self.n_layer):
                # Conv
                result = self.convs[i](result)


        result = result.squeeze(2).squeeze(2)
        result = self.fc1(result)

        return result

# 5/13: Support progressive training
# 5/13: Proxy noise generating
# 5/13: Proxy upsampling
class StyleBased_Generator(nn.Module):
    '''
    Main Module
    '''

    def __init__(self, n_fc, dim_latent, dim_input):
        super().__init__()
        # Waiting to adjust the size
        self.fcs = Intermediate_Generator(n_fc, dim_latent)
        self.convs = nn.ModuleList([
            Early_StyleConv_Block(512, dim_latent, dim_input),
            StyleConv_Block(512, 512, dim_latent),
            StyleConv_Block(512, 512, dim_latent),
            StyleConv_Block(512, 512, dim_latent),
            StyleConv_Block(512, 256, dim_latent),
            StyleConv_Block(256, 128, dim_latent),
            StyleConv_Block(128, 64, dim_latent),
            StyleConv_Block(64, 32, dim_latent),
            StyleConv_Block(32, 16, dim_latent)
        ])
        self.to_rgbs = nn.ModuleList([
            SConv2d(512, 1, 1),
            SConv2d(512, 1, 1),
            SConv2d(512, 1, 1),
            SConv2d(512, 1, 1),
            SConv2d(256, 1, 1),
            SConv2d(128, 1, 1),
            SConv2d(64, 1, 1),
            SConv2d(32, 1, 1),
            SConv2d(16, 1, 1)
        ])

    def forward(self, latent_z,
                step=0,  # Step means how many layers (count from 4 x 4) are used to train
                alpha=-1,  # Alpha is the parameter of smooth conversion of resolution):
                noise=None,  # TODO: support none noise
                mix_steps=[],  # steps inside will use latent_z[1], else latent_z[0]
                latent_w_center=None,  # Truncation trick in W
                psi=0):  # parameter of truncation
        if type(latent_z) != type([]):
            print('You should use list to package your latent_z')
            latent_z = [latent_z]
        if (len(latent_z) != 2 and len(mix_steps) > 0) or type(mix_steps) != type([]):
            print('Warning: Style mixing disabled, possible reasons:')
            print('- Invalid number of latent vectors')
            print('- Invalid parameter type: mix_steps')
            mix_steps = []

        latent_w = [self.fcs(latent) for latent in latent_z]
        batch_size = latent_w[0].size(0)

        # Truncation trick in W
        # Only usable in estimation
        if latent_w_center is not None:
            latent_w = [latent_w_center + psi * (unscaled_latent_w - latent_w_center)
                        for unscaled_latent_w in latent_w]

        # Generate needed Gaussian noise
        # 5/22: Noise is now generated by outer module
        # noise = []
        # result = 0
        # current_latent = 0
        # for i in range(step + 1):
        #     size = 4 * 2 ** i # Due to the upsampling, size of noise will grow
        #     noise.append(torch.randn((batch_size, 1, size, size), device=torch.device('cuda:0')))

        if 0 in mix_steps:
            current_latent = latent_w[1]
        else:
            current_latent = latent_w[0]

        result = self.convs[0](current_latent, noise[0])


        for i in range(1, step):
            if i in mix_steps:
                current_latent = latent_w[1]
            else:
                current_latent = latent_w[0]

            result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear',
                                                    align_corners=False)
            result = self.convs[i](result, current_latent, noise[i])


        if step == 0:
            result_prev = self.to_rgbs[step](result)
            return result_prev
        result_prev = self.to_rgbs[step-1](result)


        result_prev = nn.functional.interpolate(result_prev, scale_factor=2, mode='bilinear',
                                                            align_corners=False)

        result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear',
                                           align_corners=False)
        # to RGB current step
        result = self.convs[step](result, current_latent, noise[step])
        result = self.to_rgbs[step](result)

        result = (1 - alpha) * result_prev + alpha * result


        return result

    def center_w(self, zs):
        '''
        To begin, we compute the center of mass of W
        '''
        latent_w_center = self.fcs(zs).mean(0, keepdim=True)
        return latent_w_center


# Discriminator
# 5/13: Support progressive training
# 5/13: Add downsample module
# Component of Progressive GAN
# Reference: Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
# Progressive Growing of GANs for Improved Quality, Stability, and Variation, 1–26.
# Retrieved from http://arxiv.org/abs/1710.10196
class Discriminator(nn.Module):
    '''
    Main Module
    '''

    def __init__(self, sr=False):
        super().__init__()
        # Waiting to adjust the size
        self.from_rgbs = nn.ModuleList([
            SConv2d(1, 16, 1, sr=sr),
            SConv2d(1, 32, 1, sr=sr),
            SConv2d(1, 64, 1, sr=sr),
            SConv2d(1, 128, 1, sr=sr),
            SConv2d(1, 256, 1, sr=sr),
            SConv2d(1, 256, 1, sr=sr),
            SConv2d(1, 256, 1, sr=sr),
            SConv2d(1, 256, 1, sr=sr),
            SConv2d(1, 256, 1, sr=sr)
        ])
        self.convs = nn.ModuleList([
            ConvBlock(16, 32, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(32, 64, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(64, 128, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(128, 256, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(256, 256, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(256, 256, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(256, 256, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(256, 256, 3, 1, stride=(2, 2), sr=sr),
            ConvBlock(256, 256, 3, 1, (25, 8), 0, sr=False)
        ])

        self.fc1 = SLinear(256, 256, sr=sr)
        self.fc2 = SLinear(256, 1, sr=sr)

        self.n_layer = 9  # 9 layers network

    def forward(self, image,
                step=0,  # Step means how many layers (count from 4 x 4) are used to train
                alpha=-1):  # Alpha is the parameter of smooth conversion of resolution):


        if step == 0:
            result = self.from_rgbs[self.n_layer - 1](image)
            result = self.convs[self.n_layer - 1](result)

        else:
            # from RGB of current step
            result = self.from_rgbs[self.n_layer - 1 - step](image)
            result = self.convs[self.n_layer - 1 - step](result)


            half_res_im = nn.functional.interpolate(image, scale_factor=0.5,
                                                    mode='bilinear', align_corners=False)
            result_prev = self.from_rgbs[self.n_layer - step](half_res_im)

            result = alpha * result + (1 - alpha) * result_prev

            for i in range(self.n_layer - step, self.n_layer):
                # Conv
                result = self.convs[i](result)


        # for i in range(step, -1, -1):
        #     # Get the index of current layer
        #     # Count from the bottom layer (4 * 4)
        #     layer_index = self.n_layer - i - 1
        #
        #     # First layer, need to use from_rgb to convert to n_channel data
        #     if i == step:
        #         result = self.from_rgbs[layer_index](image)
        #
        #     # # Before final layer, do minibatch stddev
        #     # if i == 0:
        #     #     # In dim: [batch, channel(512), 4, 4]
        #     #     res_var = result.var(0, unbiased=False) + 1e-8  # Avoid zero
        #     #     # Out dim: [channel(512), 4, 4]
        #     #     res_std = torch.sqrt(res_var)
        #     #     # Out dim: [channel(512), 4, 4]
        #     #     mean_std = res_std.mean().expand(result.size(0), 1, 25, 8)
        #     #     # Out dim: [1] -> [batch, 1, 4, 4]
        #     #     result = torch.cat([result, mean_std], 1)
        #     #     # Out dim: [batch, 512 + 1, 4, 4]
        #
        #     # Conv
        #     result = self.convs[layer_index](result)
        #
        #     # Not the final layer
        #     if i > 0:
        #         # Downsample for further usage
        #         # result = nn.functional.interpolate(result, scale_factor=0.5, mode='bilinear',
        #         #                                    align_corners=False)
        #         # Alpha set, combine the result of different layers when input
        #         if i == step and 0 <= alpha < 1:
        #             result_next = self.from_rgbs[layer_index + 1](image)
        #             result_next = nn.functional.interpolate(result_next, scale_factor=0.5,
        #                                                     mode='bilinear', align_corners=False)
        #             # print(result.shape, result_next.shape)
        #             result = alpha * result + (1 - alpha) * result_next

        # Now, result is [batch, channel(512), 1, 1]
        # Convert it into [batch, channel(512)], so the fully-connetced layer
        # could process it.
        result = result.squeeze(2).squeeze(2)
        result = self.fc1(result)
        result = self.fc2(result)

        return result