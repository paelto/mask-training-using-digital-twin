import torch.nn as nn

class UNetModelv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_shape = (32,32)
        self.output_shape = (64,160)

        base_filters = 32

        self.layers = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=(2,4), stride=(2,4), padding=(0,1)),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=(2,3), stride=(2,3), padding=(0,1)),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_filters, 1, kernel_size=1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = x.squeeze(1)
        return x

if __name__ == "__main__":
    import torchsummary

    model = UNetModelv2()
    torchsummary.summary(model, model.input_shape)