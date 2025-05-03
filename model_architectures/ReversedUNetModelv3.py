import torch.nn as nn

class ReversedUNetModelv3(nn.Module):
    input_shape = (64,128)
    output_shape = (100,100)

    def __init__(self, n, f_beg, f_mid, f_end, f_fin, batch_norm):
        super().__init__()

        assert 5 <= n <= 7

        fe = []
        for i in range(n):
            fe.append(int(f_beg*(f_mid/f_beg)**(i/(n-1))))

        fd = []
        for i in range(n):
            fd.append(int(f_mid*(f_end/f_mid)**(i/n)))
        fd.append(f_fin)
        fd.append(1)

        list.reverse(fe)
        list.reverse(fd)

        encoder_layers = []

        for i in range(n+1):
            if i == n:
                encoder_layers.append(nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0))
            elif i != 0:
                encoder_layers.append(nn.MaxPool2d(kernel_size=2))

            encoder_layers.append(nn.Conv2d(fd[i], fd[i+1], kernel_size=3, padding=1))

            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(fd[i+1]))

            encoder_layers.append(nn.ReLU(inplace=True))

            encoder_layers.append(nn.Conv2d(fd[i+1], fd[i+1], kernel_size=3, padding=1))

            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(fd[i+1]))

            encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)

        pad = [1] * (n - 5) + [2] * 3 + [1]

        decoder_layers=[]

        for i in range(n-1):
            ks=2
            if i == n-5:
                ks=3
            decoder_layers.append(nn.ConvTranspose2d(fe[i], fe[i], kernel_size=ks, stride=2, padding=0))

            decoder_layers.append(nn.Conv2d(fe[i], fe[i+1], kernel_size=3, stride=1, padding=pad[i]))

            if batch_norm:
                decoder_layers.append(nn.BatchNorm2d(fe[i+1]))

            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.Conv2d(fe[i+1], fe[i+1], kernel_size=3, padding=1))

            if batch_norm:
                decoder_layers.append(nn.BatchNorm2d(fe[i+1]))

            decoder_layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decoder_layers)

        self.output_layer = nn.Conv2d(fe[n-1], 1, kernel_size=1, padding=0)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

if __name__ == "__main__":
    import torchsummary

    model = ReversedUNetModelv3(6, 10, 200, 10, 5, False)
    torchsummary.summary(model, model.input_shape)