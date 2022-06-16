
import pytorch_lightning as pl
import hydra
import torch
import os
import numpy as np
from lib.gdna_model import BaseModel
from tqdm import tqdm
import imageio
from lib.utils.render import render_mesh_dict, Renderer
import glob
from lib.dataset.datamodule import DataProcessor
from lib.model.helpers import Dict2Class
import pandas

@hydra.main(config_path="config", config_name="config")

def main(opt):

    print(opt.pretty())
    pl.seed_everything(opt.seed, workers=True)
    torch.set_num_threads(10) 

    scan_info = pandas.read_csv(hydra.utils.to_absolute_path(opt.datamodule.data_list))
    meta_info = Dict2Class({'n_samples': len(scan_info)})

    data_processor = DataProcessor(opt.datamodule)
    checkpoint_path = os.path.join('./checkpoints', 'last.ckpt')
    
    model = BaseModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        opt=opt.model, 
        meta_info=meta_info,
        data_processor=data_processor,
    ).cuda()


    renderer = Renderer(256, anti_alias=True)

    output_folder = 'videos'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    max_samples = 200

    smpl_param_zero = torch.zeros((1,86)).cuda().float()
    smpl_param_zero[:,0] = 1

    motion_folder = hydra.utils.to_absolute_path('data/aist_demo/seqs')
    motion_files = sorted(glob.glob(os.path.join(motion_folder, '*.npz')))
    smpl_param_anim = []
    for f in motion_files:
        f = np.load(f)
        smpl_params = np.zeros(86)
        smpl_params[0], smpl_params[4:76] = 1, f['pose']
        smpl_param_anim.append(torch.tensor(smpl_params))
    smpl_param_anim = torch.stack(smpl_param_anim).float().cuda()

    batch_list = []
    # prepare latent codes
    if opt.eval_mode == 'z_shape':

        idx_b = np.random.randint(0, meta_info.n_samples)
        
        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_shape_a = model.z_shapes.weight.data[idx_a]
            z_shape_b = model.z_shapes.weight.data[idx_b]
            z_detail = model.z_details.weight.data.mean(0)

            for i in range(10):

                z_shape = torch.lerp(z_shape_a, z_shape_b, i/10)


                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'smpl_params': smpl_param_zero}

                batch_list.append(batch)

    
    if opt.eval_mode == 'z_detail':
        idx_b = np.random.randint(0, meta_info.n_samples)
        
        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_detail_a = model.z_details.weight.data[idx_a]
            z_detail_b = model.z_details.weight.data[idx_b]
            z_shape = model.z_shapes.weight.data.mean(0)

            for i in range(10):

                z_detail = torch.lerp(z_detail_a, z_detail_b, i/10)


                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'smpl_params': smpl_param_zero}

                batch_list.append(batch)

    if opt.eval_mode == 'betas':
        z_shape = model.z_shapes.weight.data.mean(0)
        z_detail = model.z_details.weight.data.mean(0)

        betas = torch.cat([torch.linspace(0,-2,10),
                            torch.linspace(-2,0,10),
                            torch.linspace(0,2,10),
                            torch.linspace(2,0,10)])

        for i in range(len(betas)):
            smpl_param = smpl_param_zero.clone()
            smpl_param[:, -10] = betas[i]

            batch = {'z_shape': z_shape[None],
                    'z_detail': z_detail[None],
                    'smpl_params': smpl_param}

            batch_list.append(batch)

    if opt.eval_mode == 'thetas':
        z_shape = model.z_shapes.weight.data.mean(0)
        z_detail = model.z_details.weight.data.mean(0)

        for i in range(len(smpl_param_anim)):
            batch = {'z_shape': z_shape[None],
                     'z_detail': z_detail[None],
                     'smpl_params': smpl_param_anim[[i]]
                    }

            batch_list.append(batch)

    if opt.eval_mode == 'sample':

        z_shapes, z_details = model.sample_codes(max_samples)

        for i in range(len(z_shapes)):

            id_smpl = np.random.randint(len(smpl_param_anim))
            batch = {'z_shape': z_shapes[i][None],
                    'z_detail': z_details[i][None],
                    'smpl_params': smpl_param_anim[id_smpl][None],
                    }
            
            batch_list.append(batch)

    if opt.eval_mode == 'interp':


        idx_b = np.random.randint(0, meta_info.n_samples)
        
        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_shape_a = model.z_shapes.weight.data[idx_a]
            z_shape_b = model.z_shapes.weight.data[idx_b]

            z_detail_a = model.z_details.weight.data[idx_a]
            z_detail_b = model.z_details.weight.data[idx_b]

            for i in range(10):

                z_shape = torch.lerp(z_shape_a, z_shape_b, i/10)
                z_detail = torch.lerp(z_detail_a, z_detail_b, i/10)

                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'smpl_params': smpl_param_anim[len(batch_list)][None]}

                batch_list.append(batch)


    images_all = []
    with torch.no_grad():

        for i, batch in enumerate(tqdm(batch_list)):
        
            cond = model.prepare_cond(batch)
            batch_smpl = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, model.smpl_server)

            mesh_cano = model.extract_mesh(batch_smpl['smpl_verts_cano'], batch_smpl['smpl_tfs'], cond, res_up=4)
            mesh_def = model.deform_mesh(mesh_cano, batch_smpl['smpl_tfs'])

            img_def = render_mesh_dict(mesh_def,mode='xy',render_new=renderer)
            img_def = np.concatenate([img_def[:256,:,:], img_def[256:,:,:]],axis=1)

            images_all.append(img_def)

            if i%10 == 0:
                imageio.mimsave(os.path.join(output_folder,'%s_seed%d.mp4' % (opt.eval_mode,opt.seed)),images_all, codec='libx264')
                

if __name__ == '__main__':
    main()