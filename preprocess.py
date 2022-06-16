import os
import cv2
import torch
import kaolin
import pandas
import imageio
import numpy as np
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj
from pytorch3d.ops import sample_points_from_meshes

import sys
sys.path.append('.')

from lib.utils.render import render_pytorch3d, Renderer

class ScanProcessor():

    def __init__(self):

        self.scan_folder =  './data/THuman2.0_Release'

        self.smpl_folder =  './data/THuman2.0_smpl'

        self.scan_list = sorted(os.listdir(self.scan_folder))

        self.output_folder = './data/THuman2.0_processed'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.renderer = Renderer(256)

    def process(self, index):

        batch = {}

        scan_name = "%04d"%index

        scan_path = os.path.join(self.scan_folder,scan_name, scan_name+'.obj')

        output_folder = os.path.join(self.output_folder, scan_name)
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        batch['scan_name'] = scan_name

        pickle_path = os.path.join(self.smpl_folder, '%04d_smpl.pkl'%index)
        file = pandas.read_pickle(pickle_path)
        smpl_param = np.concatenate([np.ones( (1,1)), 
                                np.zeros( (1,3)),
                                file['global_orient'].reshape(1,-1),
                                file['body_pose'].reshape(1,-1),
                                file['betas'][:,:10]], axis=1)[0]

        batch['smpl_params'] = smpl_param

        scan_verts, scan_faces, aux = load_obj(scan_path, 
                                                device=torch.device("cuda:0"),
                                                load_textures=False)

        scan_faces = scan_faces.verts_idx.long()

        scan_verts = scan_verts - torch.tensor(file['transl']).cuda().float().expand(scan_verts.shape[0], -1)
        scan_verts = scan_verts/file['scale'][0]

        batch['scan_verts'] = scan_verts.data.cpu().numpy()
        batch['scan_faces'] = scan_faces.data.cpu().numpy()

        num_verts, num_dim = scan_verts.shape
        random_idx = torch.randint(0, num_verts, [100000, 1], device=scan_verts.device)
        pts_surf = torch.gather(scan_verts, 0, random_idx.expand(-1, num_dim))
        pts_surf += 0.01 * torch.randn(pts_surf.shape, device=scan_verts.device)
        pts_bbox = torch.rand(pts_surf.shape, device=scan_verts.device) * 2 - 1
        pts_d = torch.cat([pts_surf, pts_bbox],dim=0)
        occ_gt = kaolin.ops.mesh.check_sign(scan_verts[None], scan_faces, pts_d[None]).float().unsqueeze(-1)
        
        batch['pts_d'] = pts_d.data.cpu().numpy()
        batch['occ_gt'] = occ_gt[0].data.cpu().numpy()

        np.savez(os.path.join(output_folder, 'occupancy.npz'), **batch)


        # get surface normals 
        meshes = Meshes(verts=[scan_verts], faces=[scan_faces])
        verts, normals = sample_points_from_meshes(meshes, num_samples=100000, return_textures=False, return_normals=True)

        batch_surf = {}
        batch_surf['surface_points'] = verts[0].data.cpu().numpy()
        batch_surf['surface_normals'] = normals[0].data.cpu().numpy()

        np.savez(os.path.join(output_folder, 'surface.npz'), **batch_surf)

        # get 2D normal maps
        n_views = 18

        output_image_folder = os.path.join(output_folder, 'multi_view_256')
        if not os.path.exists(output_image_folder): os.makedirs(output_image_folder)

        for i in range(n_views):

            rot_mat = cv2.Rodrigues(np.array([0, 2*np.pi/n_views*i, 0]))[0]
            rot_mat = torch.tensor(rot_mat).cuda().float()

            meshes_new = Meshes(verts=[torch.einsum('ij,nj->ni',rot_mat,scan_verts)], faces=[scan_faces])

            image = render_pytorch3d(meshes_new, mode='n', renderer_new=self.renderer)
            imageio.imwrite(os.path.join(output_image_folder, '%04d_normal.png'%i), image)

        return 

def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]
    

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)

    args = parser.parse_args()

    processor = ScanProcessor()

    task = split( list(range( len(processor.scan_list))) , args.tot)[args.id]
    batch_list = []

    for i in tqdm(task):
        batch = processor.process(i)
    