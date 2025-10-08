from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch
import torch.nn.functional as F
import time

class TestModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='aligned')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')


        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.ifattention = opt.ifattention
        self.attention_form = opt.attention_form
        self.GNet = opt.netG
        self.spacing = opt.img_spacing
        self.loss_names = []
        self.visual_names = ["fake_E_1", "real_E_1","fake_E_2", "real_E_2","fake_E_3", "real_E_3","fake_F", "real_F"]

        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,self.ifattention,
                                      self.attention_form)

        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)

        self.real_E_1 = input['E_1'].to(self.device)
        self.real_E_2 = input['E_2'].to(self.device)
        self.real_E_3 = input['E_3'].to(self.device)
        self.real_F = input['F'].to(self.device)

        self.real_I_1 = input['I_1'].to(self.device)
        self.real_I_2 = input['I_2'].to(self.device)
        self.real_I_3 = input['I_3'].to(self.device)
        self.real_J = input['J'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):

        if self.GNet == "Swin_Unetr":
            self.fake_E_1, self.fake_E_2, self.fake_E_3 = nn.Tanh(self.netG(self.real_A,self.real_B,torch.cat((self.real_I_1, self.real_I_2,self.real_I_3), 1)) )
        else:

            self.fake_E_1, self.fake_E_2, self.fake_E_3 = self.netG(torch.cat((self.real_I_1, self.real_I_2,self.real_I_3), 1),self.real_A,self.real_B)
            self.fake_E = torch.cat([self.fake_E_1, self.fake_E_2, self.fake_E_3], dim=1)
            self.fake_E_scale = self.fake_E * (16) - 8
            self.fake_F = warp_volume(self.real_C, self.fake_E_scale, self.spacing, if_lable=False)


    def optimize_parameters(self):
        pass

def warp_volume(ct_prior, dvf_mm, spacing,if_lable=True):

    B, _, D, H, W = dvf_mm.shape
    device = dvf_mm.device

    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((xx, yy, zz), dim=0)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)


    spacing_z, spacing_y, spacing_x = spacing[2],spacing[1],spacing[0]  # in mm

    dvf_voxel = torch.zeros_like(dvf_mm)
    dvf_voxel[:, 0] = dvf_mm[:, 0] / spacing_x
    dvf_voxel[:, 1] = dvf_mm[:, 1] / spacing_y
    dvf_voxel[:, 2] = dvf_mm[:, 2] / spacing_z

    dvf_norm = torch.zeros_like(dvf_voxel)
    dvf_norm[:, 0] = dvf_voxel[:, 0] / ((W - 1) / 2)
    dvf_norm[:, 1] = dvf_voxel[:, 1] / ((H - 1) / 2)
    dvf_norm[:, 2] = dvf_voxel[:, 2] / ((D - 1) / 2)

    warped_grid = (grid + dvf_norm).permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

    if if_lable:
        image_warped = F.grid_sample(ct_prior, warped_grid, mode='nearest', padding_mode='border', align_corners=True)

    else:
        image_warped = F.grid_sample(ct_prior, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return image_warped
