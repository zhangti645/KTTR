import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactFeatureExtraction(nn.Module):
    def __init__(self):
        super(CompactFeatureExtraction, self).__init__()
        # Define layers for compact feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class FusionNetwork(nn.Module):
    def __init__(self, num_key_frames=3):
        super(FusionNetwork, self).__init__()
        self.num_key_frames = num_key_frames
        self.compact_feature_extraction = CompactFeatureExtraction()
        self.residual_network = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )
        self.gan_generator = GANGenerator()  # Assume GAN generator is defined elsewhere

    def forward(self, key_frames, res_keys):
        # Feature extraction from prediction outputs
        comp_features = [self.compact_feature_extraction(frame) for frame in key_frames]

        # Weight calculation
        distances = [torch.norm(res, p=2) for res in res_keys]
        weights = torch.softmax(torch.tensor([1/d for d in distances]), dim=0)

        # Feature fusion
        fused_feature = sum(w * f for w, f in zip(weights, comp_features))

        # Residual network processing
        fused_output = self.residual_network(fused_feature)

        # Frame generation
        reconstructed_frame = self.gan_generator(fused_output)
        return reconstructed_frame

# Example GAN generator class (simplified)
import torch
import torch.nn as nn


class GANGenerator(nn.Module):
    def __init__(self):
        super(GANGenerator, self).__init__()

        # Define the layers for the generator
        self.upsample1 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(128)

        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.upsample3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # First upsampling block with batch normalization
        x1 = self.relu(self.bn1(self.upsample1(x)))

        # Second upsampling block
        x2 = self.relu(self.bn2(self.upsample2(x1)))

        # Third upsampling block
        x3 = self.relu(self.bn3(self.upsample3(x2)))

        # Final convolution to generate the output image
        out = self.tanh(self.final_conv(x3))

        return out

