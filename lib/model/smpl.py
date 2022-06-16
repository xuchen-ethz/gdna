import torch
import hydra
import numpy as np

from lib.smpl.smplx.body_models import SMPL

class SMPLServer(torch.nn.Module):

    def __init__(self, gender='male', betas=None):
        super().__init__()

        self.smpl = SMPL(model_path=hydra.utils.to_absolute_path('lib/smpl/smpl_model'),
                         gender=gender,
                         batch_size=1,
                         use_hands=False,
                         use_feet_keypoints=False,
                         dtype=torch.float32)

        self.prev_input = None
        self.prev_output = None

        self.bone_parents = self.smpl.bone_parents.astype(int)
        self.bone_parents[0] = -1

        self.bone_ids = []
        for i in range(24): self.bone_ids.append([self.bone_parents[i], i])

        param_canonical = torch.zeros((1, 86),dtype=torch.float32)
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if betas is not None:
            param_canonical[0,-10:] = betas
        self.param_canonical = param_canonical
        smpl_output = self.forward(param_canonical, absolute=True)

        self.verts_c = smpl_output['smpl_verts']
        self.joints_c = smpl_output['smpl_jnts']
        self.tfs_c = smpl_output['smpl_tfs']
        self.tfs_c_inv = self.tfs_c.squeeze(0).inverse()
        self.weights_c = smpl_output['smpl_weights']

        param_canonical_deshaped = param_canonical.detach().clone()
        param_canonical_deshaped[0,-10:] = 0
        smpl_output_deshaped = self.forward(param_canonical, absolute=True)
        self.verts_c_deshaped = smpl_output_deshaped['smpl_verts']
        self.joints_c_deshaped = smpl_output_deshaped['smpl_jnts']
        self.tfs_c_deshaped = smpl_output_deshaped['smpl_tfs']

    def forward(self, smpl_params, displacement=None, v_template=None, absolute=False):
        """return SMPL output from params

        Args:
            smpl_params [B, 86]: smpl parameters [0-scale,1:4-trans, 4:76-thetas,76:86-betas]
            displacement [B, 6893] (optional): per vertex displacement to represent clothing. Defaults to None.

        Returns:
            verts: vertices [B,6893]
            tf_mats: bone transformations [B,24,4,4]
            weights: lbs weights [B,24]
        """


        scale, transl, thetas, betas = torch.split(smpl_params, [1, 3, 72, 10], dim=1)

        if v_template is not None:
            betas = 0*betas

        smpl_output = self.smpl.forward(betas=betas,
                                        transl=transl * 0,
                                        body_pose=thetas[:, 3:],
                                        global_orient=thetas[:, :3],
                                        return_verts=True,
                                        v_template=v_template,
                                        displacement=displacement,
                                        return_full_pose=True)

        output = {}

        verts = smpl_output.vertices.clone()
        verts = verts * (scale.unsqueeze(1)) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            param_canonical = self.param_canonical.expand(smpl_params.shape[0], -1).clone()
            param_canonical[:,-10:] = smpl_params[:,-10:]
            output_cano = self.forward(param_canonical.type_as(betas), v_template=v_template, absolute=True)

            output_cano = { k+'_cano': v for k, v in output_cano.items() }
            output.update(output_cano)

            tfs_c_inv = output_cano['smpl_tfs_cano'].inverse()
            tf_mats = torch.einsum('bnij,bnjk->bnik', tf_mats, tfs_c_inv)

        joints = smpl_output.joints.clone()
        joints = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        output.update({'smpl_verts': verts.float(),
                        'smpl_tfs': tf_mats,
                        'smpl_weights': smpl_output.weights.float(),
                        'smpl_jnts': joints.float()
                        })
        return output