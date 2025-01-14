import torch
import numpy as np
TINY_NUMBER = 1e-6
#######################################################################################################
# compute envmap from SG
#######################################################################################################
def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


def compute_envmap_pcd(lgtSGs, N=1000, upper_hemi=False):
    viewdirs = torch.randn((N, 3))
    viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)

    if upper_hemi:
        # y > 0
        viewdirs = torch.cat((viewdirs[:, 0:1], torch.abs(viewdirs[:, 1:2]), viewdirs[:, 2:3]), dim=-1)

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])

    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values

    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]

    return viewdirs.squeeze(-2), rgb

#######################################################################################################
# below are a few utility functions
#######################################################################################################
def prepend_dims(tensor, shape):
    '''
    :param tensor: tensor of shape [a1, a2, ..., an]
    :param shape: shape to prepend, e.g., [b1, b2, ..., bm]
    :return: tensor of shape [b1, b2, ..., bm, a1, a2, ..., an]
    '''
    orig_shape = list(tensor.shape)
    a = tensor.view([1] * len(shape) + orig_shape)
    b= shape + [-1] * len(orig_shape)
    tensor = tensor.view([1] * len(shape) + orig_shape).expand(shape + [-1] * len(orig_shape))
    return tensor


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    # orig impl; might be numerically unstable
    # t = torch.sqrt(lambda_val) * (1.6988 * lambda_val * lambda_val + 10.8438 * lambda_val) / (lambda_val * lambda_val + 6.2201 * lambda_val + 10.2415)

    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    # orig impl; might be numerically unstable
    # a = torch.exp(t)
    # b = torch.exp(t * cos_beta)
    # s = (a * b - 1.) / ((a - 1.) * (b + 1.))

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg(lgtSGs, metallic, roughness, diffuse_albedo, normal, viewdirs, diffuse_rgb=None, has_diffuse = True, has_specular = True):
    '''
    :param lgtSGs: [M, 7]
    :param metallic: [K, 1];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :return [..., 3]
    '''
    
    # specular_reflectance = 1 + 0 * specular_reflectance
    # roughness = 0.125 + 0 * roughness
    
    device = metallic.device
    
    metallic_output = metallic
    normals_output = 0.5 * (1 + normal) # [-1, 1] -> [0, 1]
    
    M = lgtSGs.shape[0] #128        # 64
    # K = specular_reflectance.shape[0]  # 1 

    K = 1  # 1 
    # assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1]) #the number of points or normals [28408]

    ########################################
    # specular color
    ########################################
    #### note: sanity
    # normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera       [28408, 3] -> [28408, 1, 1, 3] -> [28408, 64, 1, 3]
    normal = normal.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    # viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    viewdirs = viewdirs.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]            [28408, 3] -> [28408, 1, 1, 3] -> [28408, 64, 1, 3]

    # light; mu [0, 1, 2] (lobe axis), lambda [3] (lobe sharpness), a [4, 5, 6] (lobe amplitude)
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]        [64, 7] -> [28408, 64, 7]
    lgtSGs = lgtSGs.unsqueeze(-2).expand(dots_shape + [M, K, 7]).clone()  # [..., M, K, 7]                                      [28408, 64, 7] -> [28408, 64, 1, 7]
    # lgtSGs[:,:,:,4]=0
    # lgtSGs[:,:,:,5]=0
    # lgtSGs[:,:,:,6]=1.2
    # torch.tensor(0.1,0.3,0.5)

    #### note: sanity
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]             # 28408, 64, 1, 3 (light lobe axis)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])                                                                                  # 28408, 64, 1, 1 (light lobe sharpness)
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values                                                                   # 28408, 64, 1, 3 (light lobe amplitude)

    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    
    specular_rgb = 0

    if has_specular:
        roughness = roughness.view(-1,1) # 28408, 1
        
        # NDF
        brdfSGLobes = normal  # use normal as the brdf SG lobes                                                                     # 28408, 64, 1, 3 (material lobe axis)
        inv_roughness_pow4 = 1. / (roughness ** 4 + TINY_NUMBER)  # [K, 1] []                                                       # 28408, 1
        # brdfSGLambdas = prepend_dims(2. * inv_roughness_pow4, dots_shape + [M, ])  # [..., M, K, 1]; can be huge
        brdfSGLambdas = (2. * inv_roughness_pow4).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 1])                        # 28408, 64, 1, 1 (material lobe sharpness)
        # mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])  # [K, 1] ---> [K, 3]  
        mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape+[3])  # [K, 1] ---> [K, 3]                                          # 28408, 3
        # brdfSGMus = prepend_dims(mu_val, dots_shape + [M, ])  # [..., M, K, 3]
        brdfSGMus = mu_val.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])                                               # 28408, 64, 1, 3 (material lobe amplitude)

        # perform spherical warping
        v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)                                                        # 28408, 64, 1, 1 (material lobe axis * view dir)
        ### note: for numeric stability
        v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
        warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
        warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
        
        new_half = warpBrdfSGLobes + viewdirs # This is exactly the normal aka. brdfSGLobes!!!
        new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
        v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
        ### note: for numeric stability
        v_dot_h = torch.clamp(v_dot_h, min=0.)
        
        ORIG = True
        
        # warpBrdfSGLambdas = brdfSGLambdas / (4 * torch.abs(torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)) + TINY_NUMBER)
        
        if ORIG == True:
            warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # can be huge # WHY DID THEY USE THIS?
        else:
            warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_h + TINY_NUMBER)            
        
        warpBrdfSGMus = brdfSGMus  # [..., M, K, 3]

        # add fresnel and geometric terms; apply the smoothness assumption in SG paper
        
        # specular_reflectance = prepend_dims(specular_reflectance, dots_shape + [M, ])  # [..., M, K, 3]
        
        f0 = torch.lerp(torch.tensor(0.04, device = device).reshape([1, 1]), diffuse_albedo, metallic)
        f0 = f0.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])
        F = f0 + (1. - f0) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h) # Fresnel Term - Schlick?

        dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n> -> L.N
        ### note: for numeric stability
        dot1 = torch.clamp(dot1, min=0.)
        dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>        -> V.N
        ### note: for numeric stability
        dot2 = torch.clamp(dot2, min=0.)
        # k = (roughness + 1.) * (roughness + 1.) / 8.
        k = ((roughness + 1.) * (roughness + 1.) / 8.).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 1])
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2                                     # Geometry Shadowing Function - Schlick-GGX

        Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)   # Specular BRDF - Cook Torrance
        warpBrdfSGMus = warpBrdfSGMus * Moi

        # multiply with light sg
        final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                            warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

        # now multiply with clamped cosine, and perform hemisphere integral
        
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                        final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        # [..., M, K, 3]
        specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
        
        
        specular_rgb = specular_rgb.sum(dim=-2).sum(dim=-2)

        specular_rgb = torch.clamp(specular_rgb, min=0.)

    # ### debug
    # if torch.sum(torch.isnan(specular_rgb)) + torch.sum(torch.isinf(specular_rgb)) > 0:
    #     print('stopping here')
    #     import pdb
    #     pdb.set_trace()

    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    diffuse_rgb = None
    if has_diffuse and diffuse_rgb is None:
        black = torch.tensor(0.0, device = device).reshape([1, 1])
        c_diff = torch.lerp(diffuse_albedo, black, metallic)
        diffuse = (c_diff / np.pi).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, 1, 3])

        # multiply with light sg
        final_lobes = lgtSGLobes.narrow(dim=-2, start=0, length=1)  # [..., M, K, 3] --> [..., M, 1, 3]
        final_mus = lgtSGMus.narrow(dim=-2, start=0, length=1) * diffuse
        final_lambdas = lgtSGLambdas.narrow(dim=-2, start=0, length=1)

        # now multiply with clamped cosine, and perform hemisphere integral
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                          final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
                      final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
                      
        if has_specular:
            diffuse_rgb = (1 - F) * diffuse_rgb  # ([378330, 64, 1, 3]) * 
            
        diffuse_rgb = diffuse_rgb.sum(dim=-2).sum(dim=-2)
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)
        
    if diffuse_rgb is None:
        diffuse_rgb = 0

    # combine diffue and specular rgb, then return
    rgb = torch.clamp(specular_rgb + diffuse_rgb, min = 0.0, max = 1.0)
   
    ret = {'sg_diffuse_albedo': diffuse_albedo,
           'sg_diffuse_rgb': diffuse_rgb,
           'sg_metallic': metallic_output,
           'sg_normals': normals_output,
           'sg_rgb': rgb,
           'sg_roughness': roughness,
           'sg_specular_rgb': specular_rgb}

    return ret