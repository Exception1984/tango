import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch
from .utils import device

from .network import make_svbrdf_network
from .network import Normal_estimation_network
import open3d as o3d
import numpy as np
import ipdb

from util.o3d_utils import get_rays, per_pixel_lighting_normals

width1 = 712


def get_rays1(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = np.array([x.numpy(),y.numpy(),z.numpy()])
    look_at = np.array([-x.numpy(),-y.numpy(),-z.numpy()])
    # direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=60,
                                                              center=look_at,
                                                              eye=pos,
                                                              up=[0, -1, 0],
                                                              width_px=width1,
                                                              height_px=width1,
                                                             )
    
    return rays

def make_background(background_str, width):
    if background_str == 'black':
            background = torch.zeros(width * width, 3)
    elif background_str == 'white':
        background = torch.ones(width * width,3)
    elif background_str=='gaussian':
        background = torch.randn(width* width,3)
        background[:,1] = background[:,0]
        background[:,2] = background[:,0]
        
    return background

class NeuralStyleField(nn.Module):
    def __init__(self,  
                 material_random_pe_numfreq=0,
                 material_random_pe_sigma=12,
                 num_lgt_sgs=32,
                 max_delta_theta=np.pi/2,
                 max_delta_phi=np.pi/2,
                 normal_nerf_pe_numfreq=0,
                 normal_random_pe_numfreq=0,
                 symmetry=False,
                 radius=2.0,
                 background='black',
                 init_r_and_s=False,
                 width=512,
                 init_roughness=0.7,
                 init_specular=0.23,
                 material_nerf_pe_numfreq=0,
                 normal_random_pe_sigma=20,
                 if_normal_clamp=False,
                 num_hidden_layers = 2,
                 light_type = 'sg',
                 has_diffuse = True,
                 has_specular = True,
                 white_specular = True,
                 specular_top_k_sparsity = 1.0,
                 num_geometry_ids = 0,
                 one_net_per_component = False):
        """_summary_

        Args:
            material_random_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
            material_random_pe_sigma (int, optional): the sigma of random position encoding in svbrdf network. Defaults to 12.
            num_lgt_sgs (int, optional): the number of light SGs. Defaults to 32.
            max_delta_theta (_type_, optional): maximum offset of elevation angle whose unit is radian. Defaults to np.pi/2.
            max_delta_phi (_type_, optional): maximum offset of azimuth angle whose unit is radian. Defaults to np.pi/2.
            normal_nerf_pe_numfreq (int, optional): the number of frequencies using nerf's position encoding in normal network. Defaults to 0.
            normal_random_pe_numfreq (int, optional): the number of frequencies using random position encoding in normal network. Defaults to 0.
            symmetry (bool, optional): With this symmetry prior, the texture of the mesh will be symmetrical along the z-axis.We use this parameter in person. Defaults to False.
            radius (float, optional): the sampling raidus of camara position. Defaults to 2.0.
            background (str, optional): the background of render image.'black','white' or 'gaussian' can be selected. Defaults to 'black'.
            init_r_and_s (bool, optional): It will initialize roughness and specular if setting true. Defaults to False.
            width (int, optional): the size of render image will be [width,width]. Defaults to 512.
            init_roughness (float, optional): Initial value of roughness 0~1. Defaults to 0.7.
            init_specular (float, optional): Initial value of specular 0~1. Defaults to 0.23.
            material_nerf_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
            normal_random_pe_sigma (int, optional): the sigma of random position encoding in normal network. Defaults to 20.
            if_normal_clamp (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.svbrdf_network = make_svbrdf_network(light_type = light_type,
                                                material_random_pe_numfreq = material_random_pe_numfreq,
                                                material_random_pe_sigma = material_random_pe_sigma,
                                                dim=256,
                                                white_specular = white_specular,
                                                white_light = True,
                                                num_lgt_sgs = num_lgt_sgs,
                                                num_base_materials = 1,
                                                upper_hemi = False,                                 
                                                init_r_and_s = init_r_and_s,
                                                init_roughness=init_roughness,
                                                init_specular=init_specular,
                                                material_nerf_pe_numfreq= material_nerf_pe_numfreq,
                                                num_hidden_layers = num_hidden_layers,
                                                has_diffuse = has_diffuse,
                                                has_specular = has_specular,
                                                num_geometry_ids = num_geometry_ids,
                                                one_net_per_component = one_net_per_component)
        
        self.has_diffuse = has_diffuse
        self.has_specular = has_specular
        self.specular_top_k_sparsity = specular_top_k_sparsity
        
        self.radius = radius
        self.symmetry = symmetry
        self.width = width
        self.elev = 0.6283
        self.azim = 0.5
        self.Normal_estimation_network = Normal_estimation_network(max_delta_theta=max_delta_theta,
                                                                 max_delta_phi=max_delta_phi,
                                                                 normal_nerf_pe_numfreq=normal_nerf_pe_numfreq,
                                                                 normal_random_pe_numfreq=normal_random_pe_numfreq,
                                                                 normal_random_pe_sigma=normal_random_pe_sigma,
                                                                 if_normal_clamp = if_normal_clamp,
                                                                 num_hidden_layers = num_hidden_layers,
                                                                 num_geometry_ids = num_geometry_ids,
                                                                 one_net_per_component = one_net_per_component)
        
        self.background = make_background(background, width)

    def render_single_image(self, scene, azim, elev, radius = 2.5, geom_id_to_o3d_mesh = None):
        images = []
        normal1 = []
        normal2 = []

        roughness = []
        specular_rgb = []
        specular_albedo = []
        diffuse_rgb = []
        diffuse_albedo = []

        rays = get_rays(elev, azim, r = radius, width = 712) # get_rays1(elev, azim, r=2)
        ans = scene.cast_rays(rays)
        
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
        normal = ans['primitive_normals'][hit].reshape((-1,3))
        view_dirs = -torch.nn.functional.normalize(torch.from_numpy(rays[hit][:,3:].numpy())).to(device)
        pcd = o3d.t.geometry.PointCloud(points)
        pcd.point["normals"] = normal 
        pcd = pcd.to_legacy()
        points = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
        if self.symmetry:
            points[:,2]=torch.abs(points[:,2])
            
        # normals1 = torch.nn.functional.normalize(torch.from_numpy(np.asarray(pcd.normals))).float().to(device)
        normals1 = per_pixel_lighting_normals(scene, ans).float().to(device)

        geometry_ids = ans['geometry_ids'][hit]
        geometry_ids = torch.from_numpy(geometry_ids.numpy().astype(np.int64)).to(device)
        
        normals2 = self.Normal_estimation_network(points, normals1, geometry_ids)
        normals2 = torch.nn.functional.normalize(normals2)

        ret, _ = self.get_rbg_value(points,normals2, view_dirs, geometry_ids)
        
        del points, view_dirs
        
        hit1 = torch.from_numpy(hit.reshape(width1*width1).numpy())
        sg_rgb_values = torch.ones(width1*width1,3).float().to(device)
        sg_rgb_values[hit1] = ret['sg_rgb']
        
        image = sg_rgb_values.reshape(width1,width1,3).unsqueeze(0).detach().cpu()
        del sg_rgb_values
        image = torch.clamp(image, 0, 1)
        images.append(image)
        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        
        sg_normal1_values = torch.ones(width1*width1,3).float() #.to(device)
        sg_normal1_values[hit1]= normals1.detach().cpu()
        sg_normal1_values = sg_normal1_values.reshape(1,width1,width1,3)
        sg_normal2_values = torch.ones(width1*width1,3).float() #.to(device)
        sg_normal2_values[hit1]= normals2.detach().cpu()
        sg_normal2_values = sg_normal2_values.reshape(1,width1,width1,3)

        sg_roughness_values = torch.ones(width1*width1,1).float().to(device)
        sg_roughness_values[hit1] = ret['sg_roughness']
        sg_roughness_values = sg_roughness_values.reshape(1,width1,width1,1).detach().cpu()
        sg_roughness_values = torch.clamp(sg_roughness_values, 0, 1)

        sg_diffuse_rgb_values = torch.ones(width1*width1,3).float().to(device)
        sg_diffuse_rgb_values[hit1] = ret['sg_diffuse_rgb']
        sg_diffuse_rgb_values = sg_diffuse_rgb_values.reshape(1,width1,width1,3).detach().cpu()
        sg_diffuse_rgb_values = torch.clamp(sg_diffuse_rgb_values, 0, 1)
        
        sg_diffuse_albedo_values = torch.ones(width1 * width1, 3).float().to(device)
        sg_diffuse_albedo_values[hit1] = ret['sg_diffuse_albedo']
        sg_diffuse_albedo_values = sg_diffuse_albedo_values.reshape(1,width1,width1,3).detach().cpu()
        sg_diffuse_albedo_values = torch.clamp(sg_diffuse_albedo_values, 0, 1)

        sg_specular_rgb_values = torch.ones(width1 * width1, 3).float().to(device)
        sg_specular_rgb_values[hit1] = ret['sg_specular_rgb'] #.detach().cpu()
        sg_specular_rgb_values = sg_specular_rgb_values.reshape(1,width1,width1,3).detach().cpu()
        sg_specular_rgb_values = torch.clamp(sg_specular_rgb_values, 0, 1)
        
        sg_specular_albedo_values = torch.ones(width1 * width1, 3).float().to(device)
        spec_albedo = ret['sg_specular_albedo'] #.detach().cpu()
        sg_specular_albedo_values[hit1] = spec_albedo
        sg_specular_albedo_values = sg_specular_albedo_values.reshape(1,width1,width1,3).detach().cpu()
        sg_specular_albedo_values = torch.clamp(sg_specular_albedo_values, 0, 1)

        normal1.append(sg_normal1_values)
        normal2.append(sg_normal2_values)

        roughness.append(sg_roughness_values)
        diffuse_rgb.append(sg_diffuse_rgb_values)
        diffuse_albedo.append(sg_diffuse_albedo_values)
        specular_rgb.append(sg_specular_rgb_values)
        specular_albedo.append(sg_specular_albedo_values)
        
        normal1 = torch.cat(normal1, dim=0).permute(0, 3, 1, 2)
        normal2 = torch.cat(normal2, dim=0).permute(0, 3, 1, 2)

        roughness = torch.cat(roughness, dim=0).permute(0, 3, 1, 2)
        diffuse_rgb = torch.cat(diffuse_rgb, dim=0).permute(0, 3, 1, 2)
        diffuse_albedo = torch.cat(diffuse_albedo, dim=0).permute(0, 3, 1, 2)
        specular_rgb = torch.cat(specular_rgb, dim=0).permute(0, 3, 1, 2)
        specular_albedo = torch.cat(specular_albedo, dim=0).permute(0, 3, 1, 2)
        
        return images, normal1, normal2, roughness, diffuse_rgb, diffuse_albedo, specular_rgb, specular_albedo

    def forward(self, scene, flavor, num_views=8, std=8, center_elev=0, center_azim=0, radius = None, background = None, geom_id_to_o3d_mesh = None, target_img = None):
        if flavor == 'tango':
            if num_views>1:
                self.elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
                self.azim = torch.cat((torch.tensor([center_azim]),torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))
            if num_views==1:
                # self.elev = torch.randn(num_views) * np.pi / std + center_elev
                # self.azim += torch.rand(num_views) * 0.1
                
                #[0, 2 * pi]
                self.azim = 2 * torch.pi * torch.rand(num_views)
                
                #[-pi/2, pi/2]
                self.elev = torch.pi * torch.rand(num_views) - 0.5 * torch.pi
        else:
            num_views = len(center_azim)
            
            self.azim = torch.from_numpy(center_azim)
            self.elev = torch.from_numpy(center_elev)
            
        if radius is not None:
            self.radius = radius
            
        img_result = {}
        
        rgb = []
        diffuse_albedo = []
        diffuse_rgb = []
        specular_rgb = []
        geometry_id = []
        target_img_values = []
        
        for i in range(num_views):
            rays = get_rays(self.elev[i], self.azim[i], r=self.radius,width=self.width)
            ans = scene.cast_rays(rays)
            
            hit = ans['t_hit'].isfinite()
            
            geometry_ids = ans['geometry_ids'][hit]
            geometry_ids = torch.from_numpy(geometry_ids.numpy().astype(np.int64)).to(device)
            
            points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
            normal = ans['primitive_normals'][hit].reshape((-1,3))
            # import ipdb
            # ipdb.set_trace()
            view_dirs = -torch.nn.functional.normalize(torch.from_numpy(rays[hit][:,3:].numpy())).to(device)
            pcd = o3d.t.geometry.PointCloud(points)
            pcd.point["normals"] = normal 
            pcd = pcd.to_legacy()
            points = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
            if self.symmetry:
                points[:,2]=torch.abs(points[:,2])
                
            if geom_id_to_o3d_mesh is None:
                normals1= torch.nn.functional.normalize(torch.from_numpy(np.asarray(pcd.normals))).float().to(device)
            else:
                normals1 = per_pixel_lighting_normals(geom_id_to_o3d_mesh, ans).float().to(device)

            normals2 = self.Normal_estimation_network(points, normals1, geometry_ids)
            normals2 = torch.nn.functional.normalize(normals2)
            
            hit1 = torch.from_numpy(hit.reshape(self.width*self.width).numpy())
            target_img = target_img[i][hit1.view(self.width, self.width)][..., :3]

            ret, loss = self.get_rbg_value(points, normals2, view_dirs, geometry_ids, target_img)
            
            mask = torch.from_numpy(hit.numpy()).float().cuda().reshape(1,self.width,self.width,1)
            
            sd_geometry_ids = torch.zeros(self.width ** 2, dtype = torch.int64, device = device) - 1
            
            sg_target_values = self.background.float().to(device)
            
            if background is None:
                sg_rgb_values = self.background.float().to(device)
            elif background != self.background:
                sg_rgb_values = make_background(background, self.width).float().to(device)
                
            #     ret = {'sg_rgb': rgb,
            #    'sg_specular_rgb': specular_rgb,
            #    'sg_specular_albedo': specular_albedo,
            #    'sg_diffuse_rgb': diffuse_rgb,
            #    'sg_diffuse_albedo': diffuse_albedo,
            #    'sg_roughness': roughness}
                
            sg_rgb_values[hit1] = ret['sg_rgb']
            sd_geometry_ids[hit1] = geometry_ids
            sg_target_values[hit1] = target_img
            
            # RGB Output
            rgb.append(torch.cat((torch.clamp(sg_rgb_values.reshape(1, self.width, self.width, 3), 0, 1), mask), dim = 3))
            
            # Geomtry IDs
            geometry_id.append(sd_geometry_ids.reshape(1, self.width, self.width, 1))
            target_img_values.append(sg_target_values.reshape(1, self.width, self.width, 3))
            
            if self.has_diffuse:
                # Diffuse Albedo
                sg_diffuse_albedo_values = sg_rgb_values.clone()
                sg_diffuse_albedo_values[hit1] = ret['sg_diffuse_albedo']
                diffuse_albedo.append(torch.cat((torch.clamp(sg_diffuse_albedo_values.reshape(1, self.width, self.width, 3), 0, 1), mask), dim = 3))
                
                # Diffuse RGB
                sg_diffuse_rgb_values = sg_rgb_values.clone()
                sg_diffuse_rgb_values[hit1] = ret['sg_diffuse_rgb']
                diffuse_rgb.append(torch.cat((torch.clamp(sg_diffuse_rgb_values.reshape(1, self.width, self.width, 3), 0, 1), mask), dim = 3))
            
            # Specular RGB
            if self.has_specular:
                sg_specular_rgb_values = sg_rgb_values.clone()
                sg_specular_rgb_values[hit1] = ret['sg_specular_rgb']
                specular_rgb.append(torch.cat((torch.clamp(sg_specular_rgb_values.reshape(1, self.width, self.width, 3), 0, 1), mask), dim = 3))

        img_result['rgb'] = torch.cat(rgb, dim=0).permute(0, 3, 1, 2)
        img_result['geometry_id'] = torch.cat(geometry_id, dim = 0).permute(0, 3, 1, 2)
        img_result['target_img'] = torch.cat(target_img_values, dim = 0).permute(0, 3, 1, 2)
        
        if self.has_diffuse:
            img_result['diffuse_albedo'] = torch.cat(diffuse_albedo, dim=0).permute(0, 3, 1, 2)
            img_result['diffuse_rgb'] = torch.cat(diffuse_rgb, dim=0).permute(0, 3, 1, 2)
            
        if self.has_specular:
            img_result['specular_rgb'] = torch.cat(specular_rgb, dim=0).permute(0, 3, 1, 2)
        
        return img_result, loss
    
    def get_rbg_value(self, points, normals, view_dirs, geometry_ids, target_img = None):
        ret = { }
        sg_envmap_material = self.svbrdf_network(points, geometry_ids)
        sg_ret = self.svbrdf_network.render(sg_envmap_material, normal=normals, viewdirs=view_dirs)
        
        loss = {}
        
        loss['diffuse_albedo_loss'] = None
        loss['specular_sparsity_loss'] = None
        loss['geometry_id_to_num_pix'] = None
        
        loss['abs_mean_diff_per_geometry_id'] = None
        loss['abs_var_diff_per_geometry_id'] = None
        
        unique_geom_ids = torch.unique(geometry_ids)
        
        if self.has_diffuse:
            # Diffuse Regularization
            loss['diffuse_albedo_loss'] = {}
            loss['geometry_id_to_num_pix'] = {}
            
        if target_img is not None:
            loss['abs_mean_diff_per_geometry_id'] = {}
            loss['abs_var_diff_per_geometry_id'] = {}            
        
        for id in unique_geom_ids:
            mask = geometry_ids == id
            if self.has_diffuse:
                a = sg_envmap_material['sg_diffuse_albedo'][mask]
                num_px = len(a)
                a = torch.var(a, dim = 0)
                loss['diffuse_albedo_loss'][id.item()] = a.sum()
                loss['geometry_id_to_num_pix'][id.item()] = num_px
                
            if target_img is not None:
                rgb = sg_ret['sg_rgb']
                loss['abs_mean_diff_per_geometry_id'][id.item()] = torch.sum(torch.abs(torch.mean(rgb[mask], dim = 0) - torch.mean(target_img[mask], dim = 0)))
                loss['abs_var_diff_per_geometry_id'][id.item()] = torch.sum(torch.abs(torch.var(rgb[mask], dim = 0) - torch.var(target_img[mask], dim = 0)))
                
        if self.has_specular and self.specular_top_k_sparsity < 1.0:
            n_px = sg_ret['sg_specular_rgb'].shape[0]
            k = np.ceil(n_px * (1 - self.specular_top_k_sparsity)).astype(np.int64)
            
            px_norm = torch.sum(sg_ret['sg_specular_rgb'] ** 2, dim = -1)
            top_k_val = torch.topk(input = px_norm, k = k, largest = False)
            loss['specular_sparsity_loss'] = torch.sum(top_k_val[0])
        
        ret.update(sg_ret)
        return ret, loss