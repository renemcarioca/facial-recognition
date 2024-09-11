import numpy as np
import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, input_shape: tuple, z_dim: int, conv_blocks: int):
        super().__init__()

        self.z_dim = z_dim
        self.input_shape = input_shape
        self.conv_blocks = conv_blocks
        input_channels = input_shape[0]

        # encoder
        in_channels = input_channels
        hidden_channels = 32
        encoder_conv_list = []
        for i in range(conv_blocks):
            encoder_conv_list.append(nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1))
            encoder_conv_list.append(nn.BatchNorm2d(hidden_channels))
            encoder_conv_list.append(nn.LeakyReLU())
            if (i+1 != conv_blocks):
                in_channels = int(hidden_channels)
                hidden_channels = int(hidden_channels*2)
        self.encoder_conv = nn.Sequential(*encoder_conv_list)


        # self.encoder_conv = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        # )


        self.conv_out_size = self._get_conv_out_size(input_shape)
        self.mu = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.Dropout(0.2)
        )
        self.log_var = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.Dropout(0.2)
        )

        # decoder

        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, self.conv_out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        decoder_conv_list = []
        for _ in range(conv_blocks - 1):
            decoder_conv_list.append(nn.UpsamplingNearest2d(scale_factor=2))
            decoder_conv_list.append(nn.ConvTranspose2d(hidden_channels, int(hidden_channels/2), 3, stride=1, padding=1))
            decoder_conv_list.append(nn.BatchNorm2d(int(hidden_channels/2)))
            decoder_conv_list.append(nn.LeakyReLU())
            hidden_channels = int(hidden_channels/2)
        decoder_conv_list.append(nn.UpsamplingNearest2d(scale_factor=2))
        decoder_conv_list.append(nn.ConvTranspose2d(hidden_channels, 3, 3, stride=1, padding=1))
        decoder_conv_list.append(nn.Sigmoid())
        self.decoder_conv = nn.Sequential(*decoder_conv_list)

        # self.decoder_conv = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

    def sampling(self, mu, log_var):
            ## TODO: epsilon should be at the model's device (not CUDA)
            epsilon = torch.Tensor(np.random.normal(size=(self.z_dim), scale=1.0)).cuda()
            return mu + epsilon * torch.exp(log_var / 2)

    def forward_encoder(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size()[0], -1)
        mu_p = self.mu(x)
        log_var_p = self.log_var(x)
        return [mu_p, log_var_p]

    def forward_decoder(self, x):
        x = self.decoder_linear(x)
        x = x.view(x.size()[0], *self.conv_out_shape[1:])
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu_p, log_var_p = self.forward_encoder(x)
        x = self.sampling(mu_p, log_var_p) if self.training else mu_p
        images_p = self.forward_decoder(x)
        return [mu_p, log_var_p, images_p]

    def _get_conv_out_size(self, shape):
        out = self.encoder_conv(torch.zeros(1, *shape))
        self.conv_out_shape = out.size()
        return int(np.prod(self.conv_out_shape))

    def save(self, where):
        dump = {'z_dim': self.z_dim,
                'input_shape': self.input_shape,
                'conv_blocks': self.conv_blocks,
                'state_dict': self.state_dict()}
        torch.save(dump, where)

    def forward_no_epsilon(self, x):
        mu_p, log_var_p = self.forward_encoder(x)
        x = mu_p
        images_p = self.forward_decoder(x)
        return images_p