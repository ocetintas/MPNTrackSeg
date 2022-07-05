from torch import nn


class CNN(nn.Module):

    def __init__(self, input_dim, dims, kernel_sizes, strides, paddings, dropout_p=0.4, use_batchnorm=False):
        super(CNN, self).__init__()

        assert isinstance(dims, (list, tuple)), 'dims must be either a list or a tuple, but got {}'.format(
            type(dims))

        assert isinstance(kernel_sizes, (list, tuple)), 'kernels must be either a list or a tuple, but got {}'.format(
            type(kernel_sizes))

        assert isinstance(strides, (list, tuple)), 'strides must be either a list or a tuple, but got {}'.format(
            type(strides))

        assert isinstance(paddings, (list, tuple)), 'paddings must be either a list or a tuple, but got {}'.format(
            type(paddings))

        assert len(dims) == len(kernel_sizes) and len(kernel_sizes) == len(strides) and len(strides) == len(paddings), \
            'Number of elements mismatch between dims, kernel_sizes and strides'

        layers = []

        for i in range(len(dims)):
            layers.append(nn.Conv2d(input_dim, dims[i], kernel_size=kernel_sizes[i], stride=strides[i],
                                    padding=paddings[i]))

            if use_batchnorm and dims[i] != 1:
                layers.append(nn.BatchNorm2d(dims[i]))

            if dims[i] != 0:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p != 0 and dims[i] != 1:
                layers.append(nn.Dropout2d(p=dropout_p))

            input_dim = dims[i]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class MaskRCNNPredictor(nn.Module):
    # Mask prediction layers inspired from MaskRCNN

    def __init__(self, input_dim, dims, kernel_sizes, strides, paddings, transposed):
        super(MaskRCNNPredictor, self).__init__()

        assert isinstance(dims, (list, tuple)), 'dims must be either a list or a tuple, but got {}'.format(
            type(dims))

        assert isinstance(kernel_sizes, (list, tuple)), 'kernels must be either a list or a tuple, but got {}'.format(
            type(kernel_sizes))

        assert isinstance(strides, (list, tuple)), 'strides must be either a list or a tuple, but got {}'.format(
            type(strides))

        assert isinstance(paddings, (list, tuple)), 'paddings must be either a list or a tuple, but got {}'.format(
            type(paddings))

        assert len(dims) == len(kernel_sizes) and len(kernel_sizes) == len(strides) and len(strides) == len(paddings), \
            'Number of elements mismatch between dims, kernel_sizes and strides'

        layers = []

        for i in range(len(dims)):
            if transposed[i]:
                layers.append(nn.ConvTranspose2d(input_dim, dims[i], kernel_size=kernel_sizes[i], stride=strides[i],
                                                 padding=paddings[i]))
            else:
                layers.append(nn.Conv2d(input_dim, dims[i], kernel_size=kernel_sizes[i], stride=strides[i],
                                        padding=paddings[i]))

            if i < len(dims)-1:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

