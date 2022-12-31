from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching
from nerfacc import unpack_info, render_weight_from_alpha, accumulate_along_rays


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        update_module_step(self.texture, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_(self, rays):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        sdf_samples = []
        sdf_grad_samples = []

        def alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad = self.geometry(positions, with_grad=True, with_feature=False)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            return alpha[...,None]
        
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            sdf_samples.append(sdf)
            sdf_grad_samples.append(sdf_grad)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            rgb = self.texture(feature, t_dirs, normal)
            return rgb, alpha[...,None], sdf_grad

        with torch.no_grad():
            packed_info, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=alpha_fn,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        rgb, opacity, depth, normal_maps = custom_rendering(
            packed_info,
            t_starts,
            t_ends,
            rgb_alpha_fn=rgb_alpha_fn,
            render_bkgd=self.background_color,
        )

        sdf_samples = torch.cat(sdf_samples, dim=0)
        sdf_grad_samples = torch.cat(sdf_grad_samples, dim=0)
        opacity, depth, normal_maps = opacity.squeeze(-1), depth.squeeze(-1), normal_maps.squeeze(-1)

        rv = {
            'comp_rgb': rgb,
            'opacity': opacity,
            'depth': depth,
            'normal':normal_maps,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            rv.update({
                'sdf_samples': sdf_samples,
                'sdf_grad_samples': sdf_grad_samples
            })

        return rv

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

def custom_rendering(
    # ray marching results
    packed_info: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    # radiance field
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can be used for
    gradient-based optimization.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends`.

    Args:
        packed_info: Packed ray marching info. See :func:`ray_marching` for details.
        t_starts: Per-sample start distance. Tensor with shape (n_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (n_samples, 1).
        rgb_alpha_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 3) and opacity \
            values (N, 1). At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be \
            specified.
        early_stop_eps: Early stop threshold during trasmittance accumulation. Default: 1e-4.
        alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
        render_bkgd: Optional. Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1) and depths (n_rays, 1).

    Examples:

    .. code-block:: python

        import torch
        from nerfacc import OccupancyGrid, ray_marching, rendering

        device = "cuda:0"
        batch_size = 128
        rays_o = torch.rand((batch_size, 3), device=device)
        rays_d = torch.randn((batch_size, 3), device=device)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        # Ray marching.
        packed_info, t_starts, t_ends = ray_marching(
            rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
        )

        # Rendering.
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            # This is a dummy function that returns random values.
            rgbs = torch.rand((t_starts.shape[0], 3), device=device)
            sigmas = torch.rand((t_starts.shape[0], 1), device=device)
            return rgbs, sigmas
        colors, opacities, depths = rendering(rgb_sigma_fn, packed_info, t_starts, t_ends)

        # torch.Size([128, 3]) torch.Size([128, 1]) torch.Size([128, 1])
        print(colors.shape, opacities.shape, depths.shape)

    """
    if callable(packed_info):
        raise RuntimeError(
            "You maybe want to use the nerfacc<=0.2.1 version. For nerfacc>0.2.1, "
            "The first argument of `rendering` should be the packed ray packed info. "
            "See the latest documentation for details: "
            "https://www.nerfacc.com/en/latest/apis/rendering.html#nerfacc.rendering"
        )

    n_rays = packed_info.shape[0]
    ray_indices = unpack_info(packed_info)

    # Query sigma/alpha and color with gradients

    rgbs, alphas, sdf_grad = rgb_alpha_fn(t_starts, t_ends, ray_indices.long())
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
        rgbs.shape
    )
    assert (
        alphas.shape == t_starts.shape
    ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)
    # Rendering: compute weights and ray indices.
    weights = render_weight_from_alpha(
        packed_info, alphas, early_stop_eps, alpha_thre
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )
    normals = accumulate_along_rays(
        weights,
        ray_indices,
        values=sdf_grad,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, normals