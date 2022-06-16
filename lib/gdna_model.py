import os

import hydra
import torch
import wandb
import imageio
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from lib.model.smpl import SMPLServer
from lib.model.mesh import generate_mesh
from lib.model.sample import PointOnBones
from lib.model.generator import Generator
from lib.model.network import ImplicitNetwork
from lib.model.helpers import expand_cond, vis_images
from lib.utils.render import render_mesh_dict, weights2colors
from lib.model.deformer import skinning
from lib.model.ray_tracing import DepthModule

class BaseModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.opt = opt

        self.network = ImplicitNetwork(**opt.network)
        print(self.network)

        self.deformer = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer)

        self.generator = Generator(opt.dim_shape)
        print(self.generator)

        self.smpl_server = SMPLServer(gender='neutral')

        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        self.z_shapes = torch.nn.Embedding(meta_info.n_samples, opt.dim_shape)
        self.z_shapes.weight.data.fill_(0)

        self.z_details = torch.nn.Embedding(meta_info.n_samples, opt.dim_detail)
        self.z_details.weight.data.fill_(0)

        self.data_processor = data_processor

        if opt.stage=='fine':
            self.norm_network = ImplicitNetwork(**opt.norm_network)
            print(self.norm_network)

            if opt.use_gan:
                from lib.model.losses import GANLoss
                self.gan_loss = GANLoss(self.opt)
                print(self.gan_loss.discriminator)

        self.render = DepthModule(**self.opt.ray_tracer)


    def configure_optimizers(self):

        grouped_parameters = self.parameters()
        
        def is_included(n): 
            if self.opt.stage =='fine':
                if 'z_details' not in n and 'norm_network' not in n:
                    return False

            return True

        grouped_parameters = [
            {"params": [p for n, p in list(self.named_parameters()) if is_included(n)], 
            'lr': self.opt.optim.lr, 
            'betas':(0.9,0.999)},
        ]

        optimizer = torch.optim.Adam(grouped_parameters, lr=self.opt.optim.lr)

        if not self.opt.use_gan:
            return optimizer
        else:
            optimizer_d = torch.optim.Adam(self.gan_loss.parameters(), 
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d

    def forward(self, pts_d, smpl_tfs, smpl_verts, cond, canonical=False, canonical_shape=False, eval_mode=True, fine=False, mask=None, only_near_smpl=False):

        n_batch, n_points, n_dim = pts_d.shape

        outputs = {}        

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_d.device, dtype=torch.bool)

        # Filter based on SMPL
        if only_near_smpl:
            from kaolin.metrics.pointcloud import sided_distance
            distance, _ = sided_distance(pts_d, smpl_verts[:,::10])
            mask = mask & (distance<0.1*0.1)

        if not mask.any(): 
            return {'occ': -1000*torch.ones( (n_batch, n_points, 1), device=pts_d.device)}

        if canonical_shape:
            pts_c = pts_d 

            occ_pd, feat_pd = self.network(
                                    pts_c, 
                                    cond={'latent': cond['latent']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)
        elif canonical:
            pts_c = self.deformer.query_cano(pts_d, 
                                            {'betas': cond['betas']}, 
                                            mask=mask)

            occ_pd, feat_pd = self.network(
                                    pts_c, 
                                    cond={'latent': cond['latent']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)
        else:
            pts_c, others = self.deformer.forward(pts_d,
                                        {'betas': cond['betas'],
                                        'latent': cond['lbs']},
                                        smpl_tfs,
                                        mask=mask,
                                        eval_mode=eval_mode)

            occ_pd, feat_pd = self.network(
                                        pts_c.reshape((n_batch, -1, n_dim)), 
                                        cond={'latent': cond['latent']},
                                        mask=others['valid_ids'].reshape((n_batch, -1)),
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)

            occ_pd = occ_pd.reshape(n_batch, n_points, -1, 1)
            feat_pd = feat_pd.reshape(n_batch, n_points, -1, feat_pd.shape[-1])

            occ_pd, idx_c = occ_pd.max(dim=2)

            feat_pd = torch.gather(feat_pd, 2, idx_c.unsqueeze(-1).expand(-1, -1, 1, feat_pd.shape[-1])).squeeze(2)
            pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2)


        outputs['occ'] = occ_pd
        outputs['pts_c'] = pts_c
        outputs['weights'] = self.deformer.query_weights(pts_c,  
                                                        cond={
                                                        'betas': cond['betas'],
                                                        'latent': cond['lbs']
                                                        })
        if fine:
            outputs['norm'] = self.norm_network(pts_c, 
                                                cond={'latent': cond['detail']}, 
                                                mask=mask,
                                                input_feat=feat_pd,
                                                val_pad=1)

            smpl_tfs = expand_cond(smpl_tfs, pts_c)[mask]

            if not canonical:
                outputs['norm'][mask] = skinning(outputs['norm'][mask], outputs['weights'][mask], smpl_tfs, inverse=False, normal=True)

            outputs['norm'][mask] = outputs['norm'][mask] / torch.linalg.norm(outputs['norm'][mask],dim=-1,keepdim=True)

        return outputs


    def forward_2d(self, smpl_tfs, smpl_verts, cond, eval_mode=True, fine=True, res=256):

        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        pix_d = torch.stack([xv, yv], dim=-1).type_as(smpl_tfs)
        pix_d = pix_d.reshape(1,res*res,2)

        def occ(x, mask=None):

            outputs = self.forward(x, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, mask=mask, fine=False, only_near_smpl=True)

            if mask is not None:
                return outputs['occ'][mask].reshape(-1, 1)
            else:
                return outputs['occ']        

        pix_d = torch.stack([pix_d[...,0], -pix_d[...,1] - 0.3, torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        ray_dirs = torch.zeros_like(pix_d)
        ray_dirs[...,-1] = -1

        d = self.render(pix_d, ray_dirs, occ).detach()
        
        pix_d[...,-1] += d*ray_dirs[...,-1]

        mask = ~d.isinf()

        outputs = self.forward(pix_d, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, fine=fine, mask=mask)

        outputs['mask'] = mask

        outputs['pts_c'][~mask, :] = 1

        img = outputs['pts_c'].reshape(res,res,3).data.cpu().numpy()
        mask = outputs['mask'].reshape(res,res,1).data.cpu().numpy()

        img_mask = np.concatenate([img,mask],axis=-1)

        return img_mask

    def prepare_cond(self, batch):

        cond = {}
        cond['thetas'] =  batch['smpl_params'][:,7:-10]/np.pi
        cond['betas'] = batch['smpl_params'][:,-10:]/10.

        z_shape = batch['z_shape']            
        cond['latent'] = self.generator(z_shape)
        cond['lbs'] = z_shape
        cond['detail'] = batch['z_detail']

        return cond
    

    def training_step_coarse(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0

        reg_shape = F.mse_loss(batch['z_shape'], torch.zeros_like(batch['z_shape']))
        self.log('reg_shape', reg_shape)
        loss = loss + self.opt.lambda_reg * reg_shape
        
        reg_lbs = F.mse_loss(cond['lbs'], torch.zeros_like(cond['lbs']))
        self.log('reg_lbs', reg_lbs)
        loss = loss + self.opt.lambda_reg * reg_lbs

        outputs = self.forward(batch['pts_d'], batch['smpl_tfs'],  batch['smpl_verts'], cond, eval_mode=False, only_near_smpl=False)
        loss_bce = F.binary_cross_entropy_with_logits(outputs['occ'], batch['occ_gt'])
        self.log('train_bce', loss_bce)
        loss = loss + loss_bce

        # Bootstrapping
        num_batch = batch['pts_d'].shape[0]
        if self.current_epoch < self.opt.nepochs_pretrain:

            # Bone occupancy loss
            if self.opt.lambda_bone_occ > 0:

                pts_c, _, occ_gt, _ = self.sampler_bone.get_points(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                outputs = self.forward(pts_c, None, None, cond, canonical=True, only_near_smpl=False)
                loss_bone_occ = F.binary_cross_entropy_with_logits(outputs['occ'], occ_gt.unsqueeze(-1))
                self.log('train_bone_occ', loss_bone_occ)
                loss = loss + self.opt.lambda_bone_occ * loss_bone_occ

            # Joint weight loss
            if self.opt.lambda_bone_w > 0:
                pts_c, w_gt, _ = self.sampler_bone.get_joints(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                w_pd = self.deformer.query_weights(pts_c, {'latent': cond['lbs'], 'betas': cond['betas']*0})
                loss_bone_w = F.mse_loss(w_pd, w_gt)
                self.log('train_bone_w', loss_bone_w)
                loss = loss + self.opt.lambda_bone_w * loss_bone_w

        # Displacement loss
        pts_c_gt = self.smpl_server.verts_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1)
        pts_c = self.deformer.query_cano(batch['smpl_verts_cano'], {'betas': cond['betas']})
        loss_disp = F.mse_loss(pts_c, pts_c_gt)

        self.log('train_disp', loss_disp)
        loss = loss + self.opt.lambda_disp * loss_disp

        return loss

    def training_step_fine(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0
        
        outputs = self.forward(batch['cache_pts'], batch['smpl_tfs_img'], None, cond, canonical_shape=True, mask=batch['cache_mask'], fine=True)

        self.gan_loss_input = {
            'norm_real': batch['norm_img'],
            'norm_fake': outputs['norm'].permute(0,2,1).reshape(-1,3,self.opt.img_res,self.opt.img_res)
        }


        if batch_idx%10 == 0 and self.trainer.is_global_zero:
            img = vis_images(self.gan_loss_input)
            self.logger.experiment.log({"imgs":[wandb.Image(img)]})                  
            save_path = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), (255*img).astype(np.uint8)) 

        loss_gan, log_dict = self.gan_loss(self.gan_loss_input, self.global_step, optimizer_idx)
        for key, value in log_dict.items(): self.log(key, value)
        
        loss += self.opt.lambda_gan*loss_gan

        if optimizer_idx == 0:

            if self.opt.norm_loss_3d:           
                outputs = self.forward(batch['pts_surf'], batch['smpl_tfs'],  batch['smpl_verts'], cond, canonical=False, fine=True)
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_surf'])).mean() 
            else:
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_img'].permute(0,2,3,1).flatten(1,2)))[batch['cache_mask']].mean()
        
            self.log('loss_train/train_norm', loss_norm)
            loss += loss_norm

            reg_detail = torch.nn.functional.mse_loss(batch['z_detail'], torch.zeros_like(batch['z_detail']))
            self.log('loss_train/reg_detail', reg_detail)
            loss += self.opt.lambda_reg * reg_detail

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server, load_volume=self.opt.stage!='fine')

        batch['z_shape'] = self.z_shapes(batch['index'])
        batch['z_detail'] = self.z_details(batch['index'])

        if not self.opt.stage=='fine':
            loss = self.training_step_coarse(batch, batch_idx)
        else:
            loss = self.training_step_fine(batch, batch_idx, optimizer_idx=optimizer_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):

        # Data prep
        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server)

        batch['z_shape'] = self.z_shapes(batch['index'])
        batch['z_detail'] = self.z_details(batch['index'])

        if batch_idx == 0 and self.trainer.is_global_zero:
            with torch.no_grad(): self.plot(batch)   

    def extract_mesh(self, smpl_verts, smpl_tfs, cond, res_up=3):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False)
            return outputs['occ'].reshape(-1,1)

        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, fine=self.opt.stage=='fine', only_near_smpl=False)
        
        mesh['weights'] = outputs['weights'][0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['norm'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

        return mesh

    def deform_mesh(self, mesh, smpl_tfs):
        import copy
        # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
        mesh = copy.deepcopy(mesh)

        smpl_tfs = smpl_tfs.expand(mesh['verts'].shape[0],-1,-1,-1)
        mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs)
        
        if 'norm' in mesh:
            mesh['norm']  = skinning( mesh['norm'], mesh['weights'], smpl_tfs, normal=True)
            mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
        return mesh

    def plot(self, batch):

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        cond = self.prepare_cond(batch)

        surf_pred_cano = self.extract_mesh(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3)
        surf_pred_def  = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'])

        img_list = []
        img_list.append(render_mesh_dict(surf_pred_cano,mode='npt'))
        img_list.append(render_mesh_dict(surf_pred_def,mode='npt'))

        img_all = np.concatenate(img_list, axis=1)

        self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
        
        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), img_all)        

    def sample_codes(self, n_sample, std_scale=1):
        device = self.z_shapes.weight.device

        mean_shapes = self.z_shapes.weight.data.mean(0)
        std_shapes = self.z_shapes.weight.data.std(0)
        mean_details = self.z_details.weight.data.mean(0)
        std_details = self.z_details.weight.data.std(0)

        z_shape = torch.randn(n_sample, self.opt.dim_shape, device=device)
        z_detail = torch.randn(n_sample, self.opt.dim_detail, device=device)  

        z_shape = z_shape*std_shapes*std_scale+mean_shapes
        z_detail = z_detail*std_details*std_scale+mean_details

        return z_shape, z_detail

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)