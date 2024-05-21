import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D, Downsample2D

class PoseEncoder(nn.Module):
    def __init__(self, downscale_factor, pose_channels, channels, groups=32):
        super().__init__()
        
        # Pixel unshuffle to increase the number of channels by (downscale_factor ** 2)
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        unshuffle_output_channels = pose_channels * (downscale_factor ** 2)
        print(f"Expected conv_in input channels: {unshuffle_output_channels}")  # Debugging line

        # Ensure conv_in input channels match unshuffled output channels
        self.conv_in = nn.Conv2d(unshuffle_output_channels, channels[0], kernel_size=1)

        self.conv_pre = nn.Conv2d(channels[0], channels[0], kernel_size=1)  # 추가된 레이어

        resnets = []
        downsamplers = []
        for i in range(len(channels)):
            current_in_channels = channels[i - 1] if i > 0 else channels[0]
            out_channels = channels[i]

            # Ensure num_channels is divisible by groups
            if current_in_channels % groups != 0 or out_channels % groups != 0:
                groups = 1  # fallback to group size of 1 if not divisible

            resnets.append(ResnetBlock2D(
                in_channels=current_in_channels,
                out_channels=out_channels,
                temb_channels=None,  # no time embed
                groups=groups
            ))

            # Only add Downsample2D layer if not the last layer
            downsamplers.append(Downsample2D(
                out_channels,
                use_conv=False,
                padding=1,
                name="op"
            ) if i != len(channels) - 1 else nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(self, hidden_states):
        print(f"Input hidden_states shape: {hidden_states.shape}") 
        hidden_states = self.unshuffle(hidden_states)
        print(f"Unshuffled hidden_states shape: {hidden_states.shape}")
        
        hidden_states = self.conv_pre(hidden_states)
        print(f"Pre-conv hidden_states shape: {hidden_states.shape}")  # Debugging line

        hidden_states = self.conv_in(hidden_states)
        print(f"Convolved hidden_states shape: {hidden_states.shape}")  
        
        features = []
        for resnet, downsampler in zip(self.resnets, self.downsamplers):
            hidden_states = resnet(hidden_states, temb=None)
            features.append(hidden_states)
            hidden_states = downsampler(hidden_states)
            print(f"Resnet output hidden_states shape: {hidden_states.shape}")  

        return features
