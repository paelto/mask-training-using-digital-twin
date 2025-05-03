import torch.nn as nn

class MLPv1(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0]*input_shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, output_shape[0]*output_shape[1]),
            nn.Unflatten(1, output_shape),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    import torchsummary


    model = MLPv1(100,100), (64,128)
    torchsummary.summary(model, model.input_shape)