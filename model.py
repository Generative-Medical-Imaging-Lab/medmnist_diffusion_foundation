import gmi
import medmnist
import torch
from torch import nn
from torchvision import transforms

# huggingface diffusers unet
from diffusers import UNet2DModel


def initialize_diffusion_model(image_shape,
                               x_t_embedding_channels=16,
                               t_embedding_channels=8,
                               y_embedding_channels=8,
                               device='cuda' if torch.cuda.is_available() else 'cpu'):

    x_t_encoder = gmi.networks.SimpleCNN(input_channels=image_shape[0],
                                    output_channels=x_t_embedding_channels,
                                    hidden_channels_list=[16, 32, 16],
                                    activation=torch.nn.SiLU(),
                                    dim=2).to(device)

    t_encoder = torch.nn.Sequential(
        gmi.networks.LambdaLayer(lambda x: x.view(-1, 1)),
        gmi.networks.DenseNet((1,),
                            (t_embedding_channels,), 
                            hidden_channels_list=[16, 32, 16], 
                            activation=torch.nn.SiLU()).to(device),
        gmi.networks.LambdaLayer(lambda x: x.view(-1, t_embedding_channels, 1, 1)),
        gmi.networks.LambdaLayer(lambda x: x.repeat(1, 1, *image_shape[1:]))
    )

    y_encoder = torch.nn.Sequential(
        gmi.networks.LambdaLayer(lambda x: x.to(torch.float32)),
        gmi.networks.LambdaLayer(lambda x: x.view(-1, 1)),
        gmi.networks.DenseNet((1,),
                            (y_embedding_channels,), 
                            hidden_channels_list=[16, 32, 16], 
                            activation=torch.nn.SiLU()).to(device),
        gmi.networks.LambdaLayer(lambda x: x.view(-1, y_embedding_channels, 1, 1)),
        gmi.networks.LambdaLayer(lambda x: x.repeat(1, 1, *image_shape[1:]))
    )

    class X_0_Predictor(nn.Module):
        def __init__(self):
            super(X_0_Predictor, self).__init__()

            # self.model = gmi.networks.SimpleCNN(input_channels=x_t_embedding_channels + t_embedding_channels + y_embedding_channels,
            #                         output_channels=image_shape[0],
            #                         hidden_channels_list=[16, 32, 64, 32, 16],
            #                         activation=torch.nn.SiLU(),
            #                         dim=2).to(device)


            self.model = self.unet = UNet2DModel(
                sample_size=None,
                in_channels=x_t_embedding_channels + t_embedding_channels + y_embedding_channels,
                out_channels=image_shape[0],
                layers_per_block=4,
                norm_num_groups=8,
                block_out_channels=(32, 64),
                down_block_types=(
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                ),
            )
        
        def forward(self, x_t_embedding, t_embedding, y_embedding,t):
            assert isinstance(x_t_embedding, torch.Tensor)
            assert isinstance(t_embedding, torch.Tensor)
            assert isinstance(y_embedding, torch.Tensor) 
            x_t = torch.cat([x_t_embedding, t_embedding, y_embedding], dim=1)
            x_0_pred = self.model(x_t,t)[0]
            return x_0_pred
        
    x_0_predictor = X_0_Predictor().to(device)

    diffusion_backbone = gmi.diffusion.DiffusionBackbone(
                                x_t_encoder,
                                t_encoder,
                                x_0_predictor,
                                y_encoder=y_encoder,
                                pass_t_to_x_0_predictor=True).to(device)
    
    forward_SDE = gmi.sde.WienerProcess()

    diffusion_model = gmi.diffusion.DiffusionModel(
                                            forward_SDE,
                                            diffusion_backbone)
    
    return diffusion_model