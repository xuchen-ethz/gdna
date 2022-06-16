""" The code is based on https://github.com/lioryariv/idr with adaption. """

import torch

class NormalPerPoint():

    def __init__(self, global_sigma=1, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        if local_sigma is not None:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        sample_global = (torch.rand(batch_size, sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample

class PointOnBones():

    def __init__(self, bone_ids):
        self.bone_ids = bone_ids
        
    def get_points(self, joints, num_per_bone=5):

        num_batch, num_joints, _ = joints.shape

        samples = []
        weights = []
        bone_ids = []

        for bone_id in self.bone_ids:

            if bone_id[0]<0 or bone_id[1]<0: continue

            bone_dir = joints[:, bone_id[1]] - joints[:, bone_id[0]]

            scalars = torch.linspace(0,1,steps=num_per_bone, device=joints.device).unsqueeze(0).expand(num_batch,-1)
            scalars = (scalars + torch.randn( (num_batch,num_per_bone), device=joints.device )*0.1).clamp_(0,1)

            # b: num_batch, n: num_per_bone, i: 3-dim
            # print(joints[:, bone_id[0]].unsqueeze(1).expand(-1,scalars.shape[-1],-1).shape, scalars.shape, bone_dir.shape)
            samples.append( joints[:, bone_id[0]].unsqueeze(1).expand(-1,scalars.shape[-1],-1) + torch.einsum('bn,bi->bni', scalars, bone_dir))

            weight = torch.zeros( (num_batch,num_per_bone,num_joints), device=joints.device)
            weight[:,:,bone_id[0]] = 1
            weights.append(weight)
            bone_ids.append(torch.ones( (num_batch,num_per_bone), device=joints.device)*bone_id[0])

        samples = torch.cat(samples,1)
        weights = torch.cat(weights,1)
        bone_ids = torch.cat(bone_ids,1).long()

        probs = torch.ones( (num_batch, samples.shape[1]), device=joints.device)

        return samples, weights, probs, bone_ids


    def get_joints(self, joints):

        num_batch, num_joints, _ = joints.shape

        samples = []
        weights = []
        bone_ids = []

        for k in range(num_joints):
            samples.append( joints[:, [k]])
            weight = torch.zeros( (num_batch,1,num_joints), device=joints.device)
            weight[:,:,k] = 1
            weights.append(weight)
            bone_ids.append(torch.ones( (num_batch,1), device=joints.device)*k)

        for bone_id in self.bone_ids:

            if bone_id[0]<0 or bone_id[1]<0: continue

            samples.append( joints[:, [bone_id[1]]])

            weight = torch.zeros( (num_batch,1,num_joints), device=joints.device)
            weight[:,:,bone_id[0]] = 1
            weights.append(weight)
            bone_ids.append(torch.ones( (num_batch,1), device=joints.device)*bone_id[0])

        samples = torch.cat(samples,1)
        weights = torch.cat(weights,1)
        bone_ids = torch.cat(bone_ids,1).long()

        return samples, weights, bone_ids


