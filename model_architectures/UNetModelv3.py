import torch.nn as nn

class UNetModelv3(nn.Module):
    input_shape = (100,100)
    output_shape = (64,128)

    def __init__(self, n, f_beg, f_mid, f_end, f_fin, batch_norm):
        super().__init__()

        assert 5 <= n <= 7

        fe = [1]
        for i in range(n):
            fe.append(int(f_beg*(f_mid/f_beg)**(i/(n-1))))

        fd = []
        for i in range(n):
            fd.append(int(f_mid*(f_end/f_mid)**(i/(n-1))))
        fd.append(f_fin)

        paddings     = [1, 0, 0, 0, 1, 1, 1]
        kernal_sizes = [2, 2, 2, 3, 2, 2]

        encoder_layers = []
        
        for i in range(n):
            if i != 0:
                encoder_layers.append(nn.MaxPool2d(kernel_size=kernal_sizes[i-1], stride=2))

            encoder_layers.append(nn.Conv2d(fe[i], fe[i+1], kernel_size=3, padding=paddings[i]))

            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(fe[i+1]))

            encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(nn.Conv2d(fe[i+1], fe[i+1], kernel_size=3, padding=1))

            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(fe[i+1]))

            encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers=[]

        for i in range(n):
            if i == 0:
                decoder_layers.append(nn.ConvTranspose2d(fd[i], fd[i], kernel_size=(1,2), stride=(1,2), padding=0))
            else:
                decoder_layers.append(nn.ConvTranspose2d(fd[i], fd[i], kernel_size=2, stride=2, padding=0))

            decoder_layers.append(nn.Conv2d(fd[i], fd[i+1], kernel_size=3, padding=1))

            if batch_norm:
                decoder_layers.append(nn.BatchNorm2d(fd[i+1]))

            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.Conv2d(fd[i+1], fd[i+1], kernel_size=3, padding=1))

            if batch_norm:
                decoder_layers.append(nn.BatchNorm2d(fd[i+1]))

            decoder_layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decoder_layers)

        self.output_layer = nn.Conv2d(fd[n], 1, kernel_size=1, padding=0)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        x = x.squeeze(1)
        return x
        

if __name__ == "__main__":
    import torchsummary

    model = UNetModelv3(7, 30, 250, 30, 15, False)
    torchsummary.summary(model, model.input_shape)