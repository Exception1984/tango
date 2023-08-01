from .sg_render import prepend_dims
import torch

TINY_NUMBER = 1e-6

def render_with_point_lights(point_lights,          # n_L, 6 (n_L point lights, each x y z r g b)
                             light_vec,             # n_px, n_L, 3 -- [normalized]
                             diffuse_albedo,        # n_px, 3 (RGB)
                             specular_albedo,       # n_px, 1 (white) or 3 (RGB)
                             specular_exponent,     # n_px, 1
                             normal,                # n_px, 3 -- [normalized]
                             viewdirs,
                             has_diffuse = True,
                             has_specular = True):             # n_px, 3 -- [normalized]
    
    diffuse_albedo_orig = diffuse_albedo
    specular_albedo_orig = specular_albedo
    specular_exponent_orig = specular_exponent     # n_px, 1
    
    M = point_lights.shape[0]
    
    dots_shape = list(normal.shape[:-1])                            # n_px, n_L, 3 == 23657, 64, 3
    
    point_lights = point_lights.unsqueeze(0).expand(dots_shape + [M, 6])      # [n_L, 6] -> [1, n_L, 6] -> [n_px, n_L, 6]
    normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])                 # [n_px, 3] -> [n_px, 1, 3] -> [n_px, n_L, 3]
    viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3])             # [n_px, 3] -> [n_px, 1, 3] -> [n_px, n_L, 3]
    
    # Phong Model: k_d * I_d * (N . L) + k_s * I_s * (N . H)^s
    
    diffuse_rgb = 0
    if has_diffuse:
        # k_d * I_d * (N . L)
        diffuse_albedo = diffuse_albedo.unsqueeze(-2).expand(dots_shape + [M, 3]) # [n_px, 3] -> [n_px, 1, 3] -> [n_px, n_L, 3]
        N_dot_L = torch.clamp(torch.sum(normal * light_vec, dim = -1, keepdim = True), min = 0.) # only upper hemisphere
        diffuse_rgb = torch.clamp(torch.sum(diffuse_albedo * point_lights[..., 3:] * N_dot_L, dim = 1), min = 0.0, max = 1.0) # [n_px, n_L, 3] -> [n_px, 3]
        
    if has_specular:
        # k_s * I_s * (N . H)^s
        specular_albedo = specular_albedo.unsqueeze(-2).expand(dots_shape + [M, 3]) # [n_px, 3] -> [n_px, 1, 3] -> [n_px, n_L, 3]
        specular_exponent = specular_exponent.unsqueeze(-2).expand(dots_shape + [M, 1])
        
        H = 0.5 * (viewdirs + light_vec)
        H = H / (torch.norm(H, dim = -1, keepdim = True) + TINY_NUMBER)                                                         # [n_px, n_L, 3]
        N_dot_H = torch.clamp(torch.sum(normal * H, dim = -1, keepdim = True), min = 0.) # only upper hemisphere                # [n_px, n_L, 1]
        specular_rgb = torch.clamp(torch.sum(specular_albedo * point_lights[..., 3:] * (N_dot_H ** specular_exponent), dim = 1), min = 0.0, max = 1.0) # [n_px, n_L, 3] -> [n_px, 3]
        
    # combine diffue and specular rgb, then return
    rgb = diffuse_rgb + specular_rgb
   
    ret = {'sg_rgb': rgb,                               # n_px, 3
           'sg_specular_rgb': specular_rgb,             # n_px, 3
           'sg_specular_albedo': specular_albedo_orig,  # n_px, 1
           'sg_diffuse_rgb': diffuse_rgb,               # n_px, 3
           'sg_diffuse_albedo': diffuse_albedo_orig,    # n_px, 3
           'sg_roughness': torch.clamp(1.0 / specular_exponent_orig, 0.0, 1.0)}                   #

    return ret