import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, n_layers=3):
        super().__init__()
        
        self.n_layers = n_layers
        h_dim = 256
        self.shape_code = nn.Parameter(0.002*torch.randn(1, h_dim, 2, 8,8),requires_grad=True)

        # Upsampling 3D
        lin = nn.Linear(z_dim, h_dim*2) # activation='relu')
        setattr(self, "lin_code", lin)

        for l in range(n_layers):
            h_dim_next = max(h_dim//2, 64)
            enc = nn.Conv3d(h_dim, h_dim_next, kernel_size=3, stride=1, padding=1)
            lin = nn.Linear(z_dim,h_dim_next*2)
            h_dim = h_dim_next

            setattr(self, "enc" + str(l), enc)
            setattr(self, "lin" + str(l), lin)

    def forward(self, z):
        b,_ = z.size()

        h = self.shape_code.expand(b,-1, -1,-1,-1).clone()
        a = self.lin_code(z)
        h = actvn( adaIN(h,a) )

        for l in range(self.n_layers):
            lin = getattr(self, "lin" + str(l))
            enc = getattr(self, "enc" + str(l))

            h = F.upsample(h, scale_factor=2,mode='trilinear',align_corners=True)
            h = enc(h)
    
            if l == self.n_layers-1:
                h = actvn(h)
            else:
                h = actvn( adaIN(h,lin(z)))

        return h



def actvn(x):
    out = F.leaky_relu(x, inplace=True)
    return out

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4 or len(size) == 5)
    N, C = size[:2]

    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    if len(size)==5:
        feat_std = feat_std.unsqueeze(-1)
        feat_mean = feat_mean.unsqueeze(-1)

    return feat_mean, feat_std


def adaIN(content_feat, style_mean_std):
    assert(content_feat.size(1) == style_mean_std.size(1)/2)
    size = content_feat.size()
    b,c = style_mean_std.size()
    style_mean, style_std = style_mean_std[:,:c//2],style_mean_std[:,c//2:]

    style_mean = style_mean.unsqueeze(-1).unsqueeze(-1)
    style_std = style_std.unsqueeze(-1).unsqueeze(-1)
    if len(size)==5:
        style_mean = style_mean.unsqueeze(-1)
        style_std = style_std.unsqueeze(-1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)