
import pytorch_lightning as pl
import hydra
import torch
import os
import numpy as np
from lib.gdna_model import BaseModel
from tqdm import trange, tqdm
from lib.model.helpers import split,rectify_pose
from lib.dataset.datamodule import DataModule, DataProcessor

@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(10) 

    datamodule = DataModule(opt.datamodule)
    datamodule.setup(stage='fit')
    meta_info = datamodule.meta_info
    data_processor = DataProcessor(opt.datamodule)

    checkpoint_path = os.path.join('./checkpoints', 'last.ckpt')
    
    model = BaseModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        opt=opt.model, 
        meta_info=meta_info,
        data_processor=data_processor,
    ).cuda()

    # prepare latent codes

    batch_list = []

    output_folder = 'cache_img_dvr'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    task = split( list(range( meta_info.n_samples)), opt.agent_tot)[opt.agent_id]

    for index in tqdm(task):


        scan_info = meta_info.scan_info.iloc[index]
        f = np.load(os.path.join(meta_info.dataset_path, scan_info['id'], 'occupancy.npz') )

        batch = {'index': torch.tensor(index).long().cuda().reshape(1),
                'smpl_params': torch.tensor(f['smpl_params']).float().cuda()[None,:],
                'scan_name': scan_info['id']
                }
        
        batch_list.append(batch)

    with torch.no_grad():

        for i, batch in enumerate(tqdm(batch_list)):

            batch['z_shape'] = model.z_shapes(batch['index'])
            
            batch['z_detail'] = model.z_details(batch['index'])
            
            cond = model.prepare_cond(batch)
            scan_name = batch['scan_name']

            # smpl_batch = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, model.smpl_server)
            # mesh_cano = model.extract_mesh(smpl_batch['smpl_verts_cano'], smpl_batch['smpl_tfs'], cond, res_up=4)
            # mesh_cano['color'] = mesh_cano['pts_c'].clone()

            outputs_list = []
            smpl_param_list = []

            n = 18
            for k in trange(n):

                smpl_params = batch['smpl_params'][0].data.cpu().numpy()
                smpl_thetas = rectify_pose(smpl_params[4:76], np.array([0,2*np.pi/n*k,0]))
                
                smpl_params[4:76] = smpl_thetas
                smpl_param_list.append(smpl_params.copy())

                smpl_output = model.smpl_server(torch.tensor(smpl_params[None]).cuda().float(), absolute=False)

                img_mask = model.forward_2d(smpl_output['smpl_tfs'], 
                                          smpl_output['smpl_verts'], 
                                          cond, 
                                          eval_mode=True, 
                                          fine=False)

                outputs_list.append(img_mask)

            outputs_all = np.stack(outputs_list, axis=0)
            smpl_all = np.stack(smpl_param_list, axis=0)

            np.save(os.path.join(output_folder,'%s.npy'%scan_name),outputs_all)
            np.save(os.path.join(output_folder,'%s_pose.npy'%scan_name),smpl_all)


if __name__ == '__main__':
    main()