import torch.nn as nn

class UNetModelv4(nn.Module):
    input_shape = (100,100)
    output_shape = (64,128)

    def __init__(self, n, f_beg, f_mid, f_end, f_fin, dropout, dropout_p, norm, layer_norm_par):
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
        img_sizes_e  = [100, 48, 22, 9, 4, 2, 1]

        encoder_layers = []
        
        for i in range(n):
            if i != 0:
                encoder_layers.append(nn.MaxPool2d(kernel_size=kernal_sizes[i-1], stride=2))

            encoder_layers.append(nn.Conv2d(fe[i], fe[i+1], kernel_size=3, padding=paddings[i]))

            if norm == "batch_norm":
                encoder_layers.append(nn.BatchNorm2d(fe[i+1]))
            elif norm == "layer_norm":
                encoder_layers.append(nn.LayerNorm((fe[i+1],img_sizes_e[i],img_sizes_e[i]), elementwise_affine=layer_norm_par))
            if dropout:
                encoder_layers.append(nn.Dropout2d(dropout_p))

            encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(nn.Conv2d(fe[i+1], fe[i+1], kernel_size=3, padding=1))

            if norm == "batch_norm":
                encoder_layers.append(nn.BatchNorm2d(fe[i+1]))
            elif norm == "layer_norm":
                encoder_layers.append(nn.LayerNorm((fe[i+1],img_sizes_e[i],img_sizes_e[i]), elementwise_affine=layer_norm_par))
            if dropout:
                encoder_layers.append(nn.Dropout2d(dropout_p))

            encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers=[]

        for i in range(n):
            if i == 0:
                decoder_layers.append(nn.ConvTranspose2d(fd[i], fd[i], kernel_size=(1,2), stride=(1,2), padding=0))
            else:
                decoder_layers.append(nn.ConvTranspose2d(fd[i], fd[i], kernel_size=2, stride=2, padding=0))

            decoder_layers.append(nn.Conv2d(fd[i], fd[i+1], kernel_size=3, padding=1))

            if norm == "batch_norm":
                decoder_layers.append(nn.BatchNorm2d(fd[i+1]))
            elif norm == "layer_norm":
                decoder_layers.append(nn.LayerNorm((fd[i+1],2**(7-n+i), 2**(8-n+i)), elementwise_affine=layer_norm_par))

            if dropout:
                decoder_layers.append(nn.Dropout2d(dropout_p))

            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.Conv2d(fd[i+1], fd[i+1], kernel_size=3, padding=1))

            if norm == "batch_norm":
                decoder_layers.append(nn.BatchNorm2d(fd[i+1]))
            elif norm == "layer_norm":
                decoder_layers.append(nn.LayerNorm((fd[i+1],2**(7-n+i), 2**(8-n+i)), elementwise_affine=layer_norm_par))
            if dropout:
                decoder_layers.append(nn.Dropout2d(dropout_p))

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

    model = UNetModelv4(7, 10, 500, 10, 5, True, 0.5, "layer_norm", False)
    torchsummary.summary(model, model.input_shape)
