# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.



import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from glob import glob

from . import util
from . import renderutils as ru

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda")
                                   )
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base, random_rotate=None):
        super(EnvironmentLight, self).__init__()
        self.mtx = None
        self.random_rotate = random_rotate
        self.base = base
        self.N_maps = len(base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)

    def build_mips(self, cutoff=0.99):
        idx = np.random.randint(self.N_maps) if self.N_maps else 0
        self.specular = [self.base[idx]]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]
        self.diffuse = ru.diffuse_cubemap(self.specular[-1])
        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
        if self.random_rotate != None:
            self.build_large_envmap()

    def build_large_envmap(self):
        # print("====== build large env map ======")
        # build a large env map
        self.diffuse_large = torch.cat([self.diffuse[4], self.diffuse[0], self.diffuse[5], self.diffuse[1]]*2, dim=1)
        self.specular_large = [None] * len(self.specular)
        for idx in range(len(self.specular)):
            self.specular_large[idx] = torch.cat([
                self.specular[idx][4], self.specular[idx][0], self.specular[idx][5], self.specular[idx][1]
            ]*2, dim=1)

    def rotate_env_map(self, shift_scale):
        # rotate diffuse
        img_size = self.diffuse.shape[1]
        shift = int(img_size * shift_scale)
        for i, face_id in enumerate([4,0,5,1]):
            self.diffuse[face_id] = self.diffuse_large[:, shift + img_size * i : shift + img_size * (i+1)]

        # rotate specular
        for idx in range(len(self.specular)):
            img_size = self.specular[idx].shape[1]
            shift = int(img_size * shift_scale)
            for i, face_id in enumerate([4,0,5,1]):
                self.specular[idx][face_id] = self.specular_large[idx][:, shift + img_size * i : shift + img_size * (i+1)]

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        wo = util.safe_normalize(view_pos - gb_pos)

        if specular:
            roughness = ks[..., 1:2] # y component [1,512,512,1]
            metallic  = ks[..., 2:3] # z component [1,512,512,1]
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # rotate env map
        if self.random_rotate != None:
            if self.random_rotate > 1 or self.random_rotate < -1:
                shift_scale = np.random.rand() * 4
            else:
                shift_scale = 4 * self.random_rotate
            self.rotate_env_map(shift_scale)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube') #  [1,512,512,3]
        shaded_col = diffuse * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance

        return shaded_col * (1.0 - ks[..., 0:1]) # Modulate by hemisphere visibility

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda') * scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    return cubemap

def load_env(fn, scale=1.0, random_rotate=None):
    if os.path.isdir(fn):
        fn_list = sorted(glob(os.path.join(fn, '*4k.hdr')))
        cubemaps = [_load_env_hdr(fn, scale) for fn in fn_list]
        cubemap = torch.stack(cubemaps, dim=0)
        return EnvironmentLight(cubemap, random_rotate)
    elif os.path.splitext(fn)[1].lower() == ".hdr":
        cubemap = _load_env_hdr(fn, scale)
        cubemap = cubemap.unsqueeze(0) # in order to align with multiple env maps
        return EnvironmentLight(cubemap, random_rotate)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base[0], [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)