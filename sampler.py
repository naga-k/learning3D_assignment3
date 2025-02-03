import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        device = ray_bundle.origins.device
        batch_size = ray_bundle.origins.shape[:-1]

        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = device)
        z_vals = z_vals.expand(*batch_size, self.n_pts_per_ray).unsqueeze(-1)

        # TODO (Q1.4): Sample points from z values
        directions = ray_bundle.directions.unsqueeze(-2).expand(*batch_size, self.n_pts_per_ray, 3)
        origins = ray_bundle.origins.unsqueeze(-2).expand(*batch_size, self.n_pts_per_ray, 3)

        sample_points = origins + directions * z_vals
        
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}