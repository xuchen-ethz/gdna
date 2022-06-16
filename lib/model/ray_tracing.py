""" The code is based on https://github.com/autonomousvision/differentiable_volumetric_rendering with adaption. """

import torch
import numpy as np
import torch.nn as nn


class DepthModule(nn.Module):
    ''' Depth Module class.

    The depth module is a wrapper class for the autograd function
    DepthFunction (see below).

    Args:
        tau (float): threshold value
        n_steps (tuple): number of evaluation steps; if the difference between
            n_steps[0] and n_steps[1] is larger then 1, the value is sampled
            in the range
        n_secant_steps (int): number of secant refinement steps
        depth_range (tuple): range of possible depth values; not relevant when
            unit cube intersection is used
        method (string): refinement method (default: 'scant')
        check_cube_intersection (bool): whether to intersect rays with unit
            cube for evaluations
        max_points (int): max number of points loaded to GPU memory
    '''

    def __init__(self, tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                 depth_range=[0., 2.4], method='secant',
                 check_cube_intersection=True, max_points=3700000):
        super().__init__()
        self.tau = tau
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps
        self.depth_range = depth_range
        self.method = method
        self.check_cube_intersection = check_cube_intersection
        self.max_points = max_points

        self.calc_depth = DepthFunction.apply

    def forward(self, ray0, ray_direction, decoder, c=None, it=None,
                n_steps=None):
        ''' Calls the depth function and returns predicted depth values.

        NOTE: To avoid transformations, we assume to already have world
        coordinates and we return the d_i values of the function
            ray(d_i) = ray0 + d_i * ray_direction
        for ease of computation.
        (We can later transform the predicted points e.g. to the camera space
        to obtain the "normal" depth value as the z-axis of the transformed
        point.)

        Args:
            ray0 (tensor): ray starting points (camera center)
            ray_direction (tensor): direction of ray
            decoder (nn.Module): decoder model to evaluate points on the ray
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            n_steps (tuple): number of evaluation steps; this overwrites
                self.n_steps if not None.
        '''
        device = ray0.device
        batch_size, n_p, _ = ray0.shape
        if n_steps is None:
            n_steps = self.n_steps
        if n_steps[1] > 1:
            inputs = [ray0, ray_direction, decoder, c, n_steps,
                      self.n_secant_steps, self.tau, self.depth_range,
                      self.method, self.check_cube_intersection,
                      self.max_points]
            d_hat = self.calc_depth(*inputs)
        else:
            d_hat = torch.full((batch_size, n_p), np.inf, device=device)
        return d_hat


class DepthFunction(torch.autograd.Function):
    ''' Depth Function class.

    It provides the function to march along given rays to detect the surface
    points for the OccupancyNetwork. The backward pass is implemented using
    the analytic gradient described in the publication.
    '''
    @staticmethod
    def run_Bisection_method(d_low, d_high, n_secant_steps, ray0_masked,
                             ray_direction_masked, decoder, c, logit_tau):
        ''' Runs the bisection method for interval [d_low, d_high].

        Args:
            d_low (tensor): start values for the interval
            d_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            decoder (nn.Module): decoder model to evaluate point occupancies
            c (tensor): latent conditioned code c
            logit_tau (float): threshold value in logits
        '''
        d_pred = (d_low + d_high) / 2.
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                f_mid = decoder(p_mid, c, batchwise=False,
                                only_occupancy=True) - logit_tau
            ind_low = f_mid < 0
            d_low[ind_low] = d_pred[ind_low]
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            d_pred = 0.5 * (d_low + d_high)
        return d_pred

    @staticmethod
    def run_Secant_method(f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, decoder, c,
                          logit_tau, mask):
        ''' Runs the secant method for interval [d_low, d_high].

        Args:
            d_low (tensor): start values for the interval
            d_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            decoder (nn.Module): decoder model to evaluate point occupancies
            c (tensor): latent conditioned code c
            logit_tau (float): threshold value in logits
        '''
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):

            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked


            # prepare a batchwise format of p_mid
            p_mid_batch = torch.zeros((mask.shape[0], mask.shape[1], 3), device=mask.device)
            p_mid_batch[mask] = p_mid

            with torch.no_grad():
                f_mid = decoder(p_mid_batch, mask)[:,0] - logit_tau
            ind_low = f_mid < 0
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred

    @staticmethod
    def perform_ray_marching(ray0, ray_direction, decoder, c=None,
                             tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                             depth_range=[0., 2.4], method='secant',
                             check_cube_intersection=True, max_points=3500000):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            decoder (nn.Module): decoder model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        '''
        # Shotscuts
        batch_size, n_pts, D = ray0.shape
        device = ray0.device
        logit_tau = get_logits_from_prob(tau)
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()

        # Prepare d_proposal and p_proposal in form (b_size, n_pts, n_steps, 3)
        # d_proposal are "proposal" depth values and p_proposal the
        # corresponding "proposal" 3D points
        d_proposal = torch.linspace(
            depth_range[0], depth_range[1], steps=n_steps, device=device).view(
                1, 1, n_steps, 1)
        d_proposal = d_proposal.repeat(batch_size, n_pts, 1, 1)

        if check_cube_intersection:
            d_proposal_cube, mask_inside_cube = \
                get_proposal_points_in_unit_cube(ray0, ray_direction,
                                                 padding=0.1,
                                                 eps=1e-6, n_steps=n_steps)
            d_proposal[mask_inside_cube] = d_proposal_cube[mask_inside_cube]

        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
            ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

        # Evaluate all proposal points in parallel
        with torch.no_grad():
            output = []            

            for p_split in torch.split(p_proposal.view(batch_size, -1, 3), int(max_points / batch_size), dim=1):
                res = decoder(p_split) - logit_tau
                output.append(res)
                
            val = torch.cat(output, dim=1) 
            val = val.view(batch_size, -1, n_steps)


        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] < 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1, device=device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1, device=device).float()
        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # write c in pointwise format
        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]

        # Apply surface depth refinement step (e.g. Secant method)
        if method == 'secant' and mask.sum() > 0:
            d_pred = DepthFunction.run_Secant_method(
                f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
                ray_direction_masked, decoder, c, logit_tau, mask)
        elif method == 'bisection' and mask.sum() > 0:
            d_pred = DepthFunction.run_Bisection_method(
                d_low, d_high, n_secant_steps, ray0_masked,
                ray_direction_masked, decoder, c, logit_tau)
        else:
            d_pred = torch.ones(ray_direction_masked.shape[0], device=device)

        # for sanity
        pt_pred = torch.ones(batch_size, n_pts, 3, device=device)
        pt_pred[mask] = ray0_masked + \
            d_pred.unsqueeze(-1) * ray_direction_masked
        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts, device=device)
        d_pred_out[mask] = d_pred

        return d_pred_out, pt_pred, mask, mask_0_not_occupied

    @staticmethod
    def forward(ctx, *input):
        ''' Performs a forward pass of the Depth function.

        Args:
            input (list): input to forward function
        '''
        (ray0, ray_direction, decoder, c, n_steps, n_secant_steps, tau,
         depth_range, method, check_cube_intersection, max_points) = input[:11]

        # Get depth values
        with torch.no_grad():
            d_pred, p_pred, mask, mask_0_not_occupied = \
                DepthFunction.perform_ray_marching(
                    ray0, ray_direction, decoder, c, tau, n_steps,
                    n_secant_steps, depth_range, method, check_cube_intersection,
                    max_points)

        # Insert appropriate values for points where no depth is predicted
        d_pred[mask == 0] = np.inf
        d_pred[mask_0_not_occupied == 0] = 0

        # Save values for backward pass
        ctx.save_for_backward(ray0, ray_direction, d_pred, p_pred, c)
        ctx.decoder = decoder
        ctx.mask = mask

        return d_pred

    @staticmethod
    def backward(ctx, grad_output):
        ''' Performs the backward pass of the Depth function.

        We use the analytic formula derived in the main publication for the
        gradients. 

        Note: As for every input a gradient has to be returned, we return
        None for the elements which do no require gradients (e.g. decoder).

        Args:
            ctx (Pytorch Autograd Context): pytorch autograd context
            grad_output (tensor): gradient outputs
        '''
        ray0, ray_direction, d_pred, p_pred, c = ctx.saved_tensors
        decoder = ctx.decoder
        mask = ctx.mask
        eps = 1e-3

        with torch.enable_grad():
            p_pred.requires_grad = True
            f_p = decoder(p_pred, c, only_occupancy=True)
            f_p_sum = f_p.sum()
            grad_p = torch.autograd.grad(f_p_sum, p_pred, retain_graph=True)[0]
            grad_p_dot_v = (grad_p * ray_direction).sum(-1)

            if mask.sum() > 0:
                grad_p_dot_v[mask == 0] = 1.
                # Sanity
                grad_p_dot_v[abs(grad_p_dot_v) < eps] = eps
                grad_outputs = -grad_output.squeeze(-1)
                grad_outputs = grad_outputs / grad_p_dot_v
                grad_outputs = grad_outputs * mask.float()

            # Gradients for latent code c
            if c is None or c.shape[-1] == 0 or mask.sum() == 0:
                gradc = None
            else:
                gradc = torch.autograd.grad(f_p, c, retain_graph=True,
                                            grad_outputs=grad_outputs)[0]

            # Gradients for network parameters phi
            if mask.sum() > 0:
                # Accumulates gradients weighted by grad_outputs variable
                grad_phi = torch.autograd.grad(
                    f_p, [k for k in decoder.parameters()],
                    grad_outputs=grad_outputs, retain_graph=True)
            else:
                grad_phi = [None for i in decoder.parameters()]

        # Return gradients for c, z, and network parameters and None
        # for all other inputs
        out = [None, None, None, gradc, None, None, None, None, None,
               None, None] + list(grad_phi)
        return tuple(out)


def get_proposal_points_in_unit_cube(ray0, ray_direction, padding=0.1,
                                     eps=1e-6, n_steps=40):
    ''' Returns n_steps equally spaced points inside the unit cube on the rays
    cast from ray0 with direction ray_direction.
    This function is used to get the ray marching points {p^ray_j} for a given
    camera position ray0 and
    a given ray direction ray_direction which goes from the camera_position to
    the pixel location.
    NOTE: The returned values d_proposal are the lengths of the ray:
        p^ray_j = ray0 + d_proposal_j * ray_direction
    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
        n_steps (int): number of steps
    '''

    p_intervals, d_intervals, mask_inside_cube = \
        check_ray_intersection_with_unit_cube(ray0, ray_direction, padding,
                                              eps)
    d_proposal = d_intervals[:, :, 0].unsqueeze(-1) + \
        torch.linspace(0, 1, steps=n_steps).type_as(ray0).view(1, 1, -1) * \
        (d_intervals[:, :, 1] - d_intervals[:, :, 0]).unsqueeze(-1)
    d_proposal = d_proposal.unsqueeze(-1)

    return d_proposal, mask_inside_cube

def get_prob_from_logits(logits):
    ''' Returns probabilities for logits
    Args:
        logits (tensor): logits
    '''
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    return probs


def get_logits_from_prob(probs, eps=1e-4):
    ''' Returns logits for probabilities.
    Args:
        probs (tensor): probability tensor
        eps (float): epsilon value for numerical stability
    '''
    probs = np.clip(probs, a_min=eps, a_max=1-eps)
    logits = np.log(probs / (1 - probs))
    return logits



def check_ray_intersection_with_unit_cube(ray0, ray_direction, padding=0.1,
                                          eps=1e-6):
    ''' Checks if rays ray0 + d * ray_direction intersect with unit cube with
    padding padding.
    It returns the two intersection points as well as the sorted ray lengths d.
    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
    '''
    batch_size, n_pts, _ = ray0.shape
    device = ray0.device

    # calculate intersections with unit cube (< . , . >  is the dot product)
    # <n, x - p> = <n, ray0 + d * ray_direction - p_e> = 0
    # d = - <n, ray0 - p_e> / <n, ray_direction>

    # Get points on plane p_e
    p_distance = 0.5 + padding/2
    p_e = torch.ones(batch_size, n_pts, 6, device=device) * p_distance
    p_e[:, :, 3:] *= -1.

    # Calculate the intersection points with given formula
    nominator = p_e - ray0.repeat(1, 1, 2)
    denominator = ray_direction.repeat(1, 1, 2)
    d_intersect = nominator / denominator
    p_intersect = ray0.unsqueeze(-2) + d_intersect.unsqueeze(-1) * \
        ray_direction.unsqueeze(-2)

    # Calculate mask where points intersect unit cube
    p_mask_inside_cube = (
        (p_intersect[:, :, :, 0] <= p_distance + eps) &
        (p_intersect[:, :, :, 1] <= p_distance + eps) &
        (p_intersect[:, :, :, 2] <= p_distance + eps) &
        (p_intersect[:, :, :, 0] >= -(p_distance + eps)) &
        (p_intersect[:, :, :, 1] >= -(p_distance + eps)) &
        (p_intersect[:, :, :, 2] >= -(p_distance + eps))
    ).cpu()

    # Correct rays are these which intersect exactly 2 times
    mask_inside_cube = p_mask_inside_cube.sum(-1) == 2

    # Get interval values for p's which are valid
    p_intervals = p_intersect[mask_inside_cube][p_mask_inside_cube[
        mask_inside_cube]].view(-1, 2, 3)
    p_intervals_batch = torch.zeros(batch_size, n_pts, 2, 3, device=device)
    p_intervals_batch[mask_inside_cube] = p_intervals

    # Calculate ray lengths for the interval points
    d_intervals_batch = torch.zeros(batch_size, n_pts, 2, device=device)
    norm_ray = torch.norm(ray_direction[mask_inside_cube], dim=-1)
    d_intervals_batch[mask_inside_cube] = torch.stack([
        torch.linalg.norm(p_intervals[:, 0] -
                   ray0[mask_inside_cube], dim=-1) / norm_ray,
        torch.linalg.norm(p_intervals[:, 1] -
                   ray0[mask_inside_cube], dim=-1) / norm_ray,
    ], dim=-1)

    # Sort the ray lengths
    d_intervals_batch, indices_sort = d_intervals_batch.sort()
    p_intervals_batch = p_intervals_batch[
        torch.arange(batch_size).view(-1, 1, 1),
        torch.arange(n_pts).view(1, -1, 1),
        indices_sort
    ]

    return p_intervals_batch, d_intervals_batch, mask_inside_cube
