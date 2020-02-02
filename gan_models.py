import torch
import torch.nn as nn

# Input: (3, 128, 128)
# Output: Scalar value indicating probability that input comes from a standard normal distribution of the dataset

class Descriminator(nn.Module):
    def __init__(self, lr, beta):
        super(Descriminator, self).__init__()
        
        in_channels = 3
        feature_maps = 64
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*4, feature_maps*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*8, feature_maps*16, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*16, feature_maps*16, 3, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feature_maps*16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(feature_maps*8, 1, 4, 1, bias=False),
            nn.Conv2d(feature_maps*16, 1, 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.apply(self.init_weights)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta, 0.999))
        
        
    def forward(self, x):
        x = self.seq(x)
        
        return x
    
    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)



class Generator(nn.Module):
    def __init__(self, lr, beta):
        super(Generator, self).__init__()
        
        z_size = 100
        feature_maps = 64
        num_channels = 3
        
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(z_size, feature_maps*16, 5, 1, bias=False),
            nn.BatchNorm2d(feature_maps*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_maps*16, feature_maps*16, 4, 1, bias=False),
            nn.BatchNorm2d(feature_maps*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_maps*16, feature_maps*8, 4, 2, 1, bias=False),
            # nn.ConvTranspose2d(z_size, feature_maps*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_maps*8, feature_maps*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_maps*2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_maps, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.apply(self.init_weights)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta, 0.999))
#         self.to(device)
#         self.check_output()
        
    def forward(self, x):
        x = self.seq(x)
        return x
    
    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def check_output(self):
#         z = torch.randn(10,100).to(device)
        z = torch.zeros(1, 100, 1, 1)
        output = self.seq(z)
        print(output.shape)

# def test_output():
#     G_net = Generator(lr=0.001, beta=0.5)
#     D_net = Descriminator(lr=0.001, beta=0.5)

#     dummy_input = torch.zeros(1, 3, 256, 256)
#     res = D_net.forward(dummy_input)
#     print(res.shape)
#     G_net.check_output()

# test_output()