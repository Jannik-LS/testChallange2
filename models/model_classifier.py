import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, downsample_layer=None):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                                     stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(output_channels)

        self.conv_layer2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(output_channels)

        self.activation = nn.ReLU(inplace=True)
        self.downsample_layer = downsample_layer

    def forward(self, input_tensor):
        residual = input_tensor
        out = self.activation(self.batchnorm1(self.conv_layer1(input_tensor)))
        out = self.batchnorm2(self.conv_layer2(out))

        if self.downsample_layer is not None:
            residual = self.downsample_layer(input_tensor)

        out += residual
        return self.activation(out)

class ResNet14(nn.Module):
    def __init__(self, input_channels=1, number_classes=50):
        super().__init__()
        self.current_channels = 64

        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=7,
                                      stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(output_channels=64, num_blocks=2)
        self.stage2 = self._make_stage(output_channels=128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(output_channels=256, num_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, number_classes)

    def _make_stage(self, output_channels, num_blocks, stride=1):
        downsample_layer = None
        if stride != 1 or self.current_channels != output_channels:
            downsample_layer = nn.Sequential(
                nn.Conv2d(self.current_channels, output_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )

        layers = [ResidualBlock(self.current_channels, output_channels, stride, downsample_layer)]
        self.current_channels = output_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(output_channels, output_channels))

        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        out = self.activation(self.initial_bn(self.initial_conv(input_tensor)))
        out = self.initial_pool(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
