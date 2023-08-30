import torch
import torch.nn as nn
import numpy as np
from .embedder import get_embedder,FourierFeatureTransform

from .sg_render import render_with_sg
from .point_lights_render import render_with_point_lights

TINY_NUMBER = 1.0e-6

### uniformly distribute points on a sphere
def fibonacci_sphere(samples = 1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    '''
    
    if samples == 1:
        return np.array([0.0, 1.0, 0.0])
    
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    
    return points

def compute_energy(lgtSGs): # [M, 7]; lobe + lambda + mu -> mu [0, 1, 2] (lobe axis), lambda [3] (lobe sharpness), a [4:7] (lobe amplitude)
    lgtLambda = torch.abs(lgtSGs[:, 3:4])          # [M, 1] # lobe sharpness
    lgtMu = torch.abs(lgtSGs[:, 4:])               # [M, 3] # lobe amplitude
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy

def make_svbrdf_network(
    light_type = 'sg',
    material_random_pe_numfreq=128, 
    dim=256,
    material_random_pe_sigma=1,
    white_specular = True,
    white_light=False,
    num_lgt_sgs=32,
    num_base_materials=1,
    upper_hemi=False,
    init_r_and_s = False,
    init_roughness=0.7,
    init_specular=0.23,
    material_nerf_pe_numfreq=0,
    num_hidden_layers = 2,
    has_diffuse = True,
    has_specular = True,
    num_geometry_ids = None,
    one_net_per_component = False):
    
    if light_type == 'sg':
        return svbrdf_sg_network(
            material_random_pe_numfreq, 
            dim,
            material_random_pe_sigma,
            white_specular,
            white_light,
            num_lgt_sgs,
            num_base_materials,
            upper_hemi,
            init_r_and_s,
            init_roughness,
            init_specular,
            material_nerf_pe_numfreq,
            num_hidden_layers,
            has_diffuse,
            has_specular,
            num_geometry_ids,
            one_net_per_component)
    elif light_type == 'point':
        return svbrdf_point_network(
            num_lgt_sgs,
            num_geometry_ids,
            white_specular,
            white_light,
            has_diffuse,
            has_specular)
    else:
        raise NotImplementedError("No light_type {}".format(light_type))

def get_activation_function(activation):
    if activation == str.lower('relu'):
        return nn.ReLU()
    elif activation == str.lower('prelu'):
        return nn.PReLU()

class ResNetBlockPlusActivation(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        
        if isinstance(activation, str):
            self.activation_1 = get_activation_function(activation)
            self.activation_2 = get_activation_function(activation)
        else:
            self.activation_1 = activation
            self.activation_2 = activation
        
        self.net = nn.Sequential(nn.Linear(dim, dim), self.activation_1, nn.Linear(dim, dim))
        
    def forward(self, x):
        x_orig = x
        x = self.net(x)
        
        return self.activation_2(x + x_orig)

class LinearPlusActivation(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        
        if isinstance(activation, str):
            self.activation = get_activation_function(activation)
        else:
            self.activation = activation
        
        self.net = nn.Sequential(nn.Linear(dim, dim), self.activation)
        
    def forward(self, x):
        return self.net(x)

class svbrdf_network(nn.Module):
    def __init__(self, 
                 material_random_pe_numfreq=128, 
                 dim=256,
                 material_random_pe_sigma=1,
                 white_specular = True,
                 white_light=False,
                 num_lgt_sgs=32,
                 num_base_materials=1,
                 upper_hemi=False,
                 init_r_and_s = False,
                 init_roughness=0.7,
                 init_specular=0.23,
                 material_nerf_pe_numfreq=0,
                 num_hidden_layers = 2,
                 has_diffuse = True,
                 has_specular = True,
                 num_geometry_ids = 1,
                 one_net_per_component = False):
        """_summary_

        Args:
            material_random_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 128.
            dim (int, optional): Dimension of the MLP layer. Defaults to 256.
            material_random_pe_sigma (int, optional): the sigma of random position encoding in svbrdf network. Defaults to 1.
            white_specular (bool, optional): _description_. Defaults to False.
            white_light (bool, optional): _description_. Defaults to False.
            num_lgt_sgs (int, optional): the number of light SGs. Defaults to 32.
            num_base_materials (int, optional): number of BRDF SG. Defaults to 1.
            upper_hemi (bool, optional): check if lobes are in upper hemisphere. Defaults to False.
            fix_specular_albedo (bool, optional): _description_. Defaults to False.
            init_r_and_s (bool, optional): It will initialize roughness and specular if setting true. Defaults to False.
            init_roughness (float, optional): Initial value of roughness 0~1. Defaults to 0.7.
            init_specular (float, optional): Initial value of specular 0~1. Defaults to 0.23.
            material_nerf_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
        """
        super().__init__()
        
        self.has_diffuse = has_diffuse
        self.has_specular = has_specular
        self.one_net_per_component = one_net_per_component
        
        self.activate = nn.PReLU()
        # self.activation_type = 'prelu'
        self.material_nerf_pe_numfreq = material_nerf_pe_numfreq
        
        if material_nerf_pe_numfreq > 0:       
            self.embed_fn, input_dim = get_embedder(material_nerf_pe_numfreq)
            
        init_roughness = torch.tensor(init_roughness)
        init_roughness = torch.arctanh(init_roughness*2-1)
        
        init_specular = torch.tensor(init_specular)
        init_specular = torch.arctanh(init_specular*2-1)
        
        specular_output_dim = 3
        if white_specular:
            specular_output_dim = 1
        self.specular_output_dim = specular_output_dim
        
        block_type = ResNetBlockPlusActivation # LinearPlusActivation
        
        def make_mlp_base():
            base_layers = []      
            if material_random_pe_numfreq >0:
                base_layers.append(FourierFeatureTransform(3, material_random_pe_numfreq, material_random_pe_sigma))
                base_layers.append(nn.Sequential(nn.Linear(material_random_pe_numfreq*2+3, dim), self.activate))
            elif material_nerf_pe_numfreq>0:       
                # self.embed_fn, input_dim = get_embedder(material_nerf_pe_numfreq)
                base_layers.append(nn.Sequential(nn.Linear(input_dim, 256), self.activate))
            
            for i in range(1):
                base_layers.append(nn.Sequential(nn.Linear(dim, dim), self.activate))
                return nn.ModuleList(base_layers)
        
        def make_mlp_diffuse():
            diffuse_layers=[]
            for i in range(num_hidden_layers):
                diffuse_layers.append(block_type(dim, self.activate))
            diffuse_layers.append(nn.Linear(dim, 3))
            diffuse_layers.append(nn.Tanh())
            return nn.ModuleList(diffuse_layers)
            
        def make_mlp_roughness():
            roughness_layers=[]
            for i in range(num_hidden_layers): #num_hidden_layers
                roughness_layers.append(block_type(dim, self.activate))
        
            a = nn.Linear(dim, 1)
            if init_r_and_s:
                torch.nn.init.constant_(a.weight, 0)
                torch.nn.init.constant_(a.bias, init_roughness)
            roughness_layers.append(a)
            roughness_layers.append(nn.Tanh())
            return nn.ModuleList(roughness_layers)
        
        def make_mlp_specular():
            specular_layers = []
            for i in range(num_hidden_layers): # num_hidden_layers
                specular_layers.append(block_type(dim, self.activate))
                
            b=nn.Linear(dim, specular_output_dim)
            
            if init_r_and_s:
                torch.nn.init.constant_(b.weight, 0)
                torch.nn.init.constant_(b.bias, init_specular)
            specular_layers.append(b)
            specular_layers.append(nn.Tanh())
            return nn.ModuleList(specular_layers)
            
        if one_net_per_component:
            self.mlp_base = nn.ModuleList([make_mlp_base() for _ in range(num_geometry_ids)])
        else:
            self.mlp_base = make_mlp_base()
            
        if self.has_diffuse:
            if one_net_per_component:
                self.mlp_diffuse = nn.ModuleList([make_mlp_diffuse() for _ in range(num_geometry_ids)])
            else:
                self.mlp_diffuse = make_mlp_diffuse()
        
        if self.has_specular:
            if one_net_per_component:
                self.mlp_roughness = nn.ModuleList([make_mlp_roughness() for _ in range(num_geometry_ids)])
                self.mlp_specular = nn.ModuleList([make_mlp_specular() for _ in range(num_geometry_ids)])
            else:
                self.mlp_roughness = make_mlp_roughness()
                self.mlp_specular = make_mlp_specular()

        self.numLgtSGs = num_lgt_sgs
        self.numBrdfSGs = num_base_materials
        self.white_light = white_light
        self.upper_hemi = upper_hemi
        
        self.create_lights()

    def create_lights(self):
        raise NotImplementedError()
    
    def forward(self, points, geometry_id): # = None):
        ret = {}
        ret['sg_diffuse_albedo'] = None
        ret['sg_roughness'] = None
        ret['sg_specular_reflectance'] = None
        
        x = points
        if self.material_nerf_pe_numfreq > 0:
            x = self.embed_fn(x)
            
        if self.one_net_per_component:
            geom_ids = torch.unique(geometry_id).detach().cpu().numpy()
            x = {geom_id: x[geometry_id == geom_id] for geom_id in geom_ids}
        
        if self.one_net_per_component:
            for geom_id in geom_ids:
                for layer in self.mlp_base[geom_id]:
                    x[geom_id] = layer(x[geom_id])
        else:
            for layer in self.mlp_base:
                x = layer(x)

        if self.has_specular:
            if self.one_net_per_component:
                roughness = {k:v for k, v in x.items()}
                specular_reflectance = {k:v for k, v in x.items()}
                
                for geom_id in geom_ids:
                    for layer in self.mlp_roughness[geom_id]:
                        roughness[geom_id] = layer(roughness[geom_id])
                    roughness[geom_id] = (roughness[geom_id] + 1) / 2
                    
                    for layer in self.mlp_specular[geom_id]:
                        specular_reflectance[geom_id] = layer(specular_reflectance[geom_id])
                    specular_reflectance[geom_id] = (specular_reflectance[geom_id] + 1) / 2
                    
                spec_output = torch.zeros((points.shape[0], self.specular_output_dim), dtype = points.dtype, device = points.device)
                roug_output = torch.zeros((points.shape[0], 1), dtype = points.dtype, device = points.device)
                for geom_id in geom_ids:
                    spec_output[geom_id == geometry_id] = specular_reflectance[geom_id]
                    roug_output[geom_id == geometry_id] = roughness[geom_id]
                specular_reflectance = spec_output
                roughness = roug_output
            else:
                roughness = x
                specular_reflectance = x
                for layer in self.mlp_roughness:
                    roughness = layer(roughness)
                roughness = (roughness + 1) / 2
            
                for layer in self.mlp_specular:
                    specular_reflectance = layer(specular_reflectance)
                specular_reflectance = (specular_reflectance + 1) / 2
            
            ret['sg_specular_reflectance'] = specular_reflectance
            ret['sg_roughness'] = roughness
            
        if self.has_diffuse:
            if self.one_net_per_component:
                diffuse_albedo = {k:v for k, v in x.items()}
                for geom_id in geom_ids:
                    for layer in self.mlp_diffuse[geom_id]:
                        diffuse_albedo[geom_id] = layer(diffuse_albedo[geom_id])
                    diffuse_albedo[geom_id] = (diffuse_albedo[geom_id] + 1) / 2
                diff_output = torch.zeros((points.shape[0], 3), dtype = points.dtype, device = points.device)
                for geom_id in geom_ids:
                    diff_output[geom_id == geometry_id] = diffuse_albedo[geom_id]
                diffuse_albedo = diff_output
            else:
                diffuse_albedo = x
                for layer in self.mlp_diffuse:
                    diffuse_albedo = layer(diffuse_albedo)
                diffuse_albedo = (diffuse_albedo + 1) / 2
            
            ret['sg_diffuse_albedo'] = diffuse_albedo
        
        return ret

class svbrdf_sg_network(svbrdf_network):
    def __init__(self, 
                 material_random_pe_numfreq=128, 
                 dim=256,
                 material_random_pe_sigma=1,
                 white_specular = True,
                 white_light=False,
                 num_lgt_sgs=32,
                 num_base_materials=1,
                 upper_hemi=False,
                 init_r_and_s = False,
                 init_roughness=0.7,
                 init_specular=0.23,
                 material_nerf_pe_numfreq=0,
                 num_hidden_layers = 2,
                 has_diffuse = True,
                 has_specular = True,
                 num_geometry_ids = 1,
                 one_net_per_component = False):
        """_summary_

        Args:
            material_random_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 128.
            dim (int, optional): Dimension of the MLP layer. Defaults to 256.
            material_random_pe_sigma (int, optional): the sigma of random position encoding in svbrdf network. Defaults to 1.
            white_specular (bool, optional): _description_. Defaults to False.
            white_light (bool, optional): _description_. Defaults to False.
            num_lgt_sgs (int, optional): the number of light SGs. Defaults to 32.
            num_base_materials (int, optional): number of BRDF SG. Defaults to 1.
            upper_hemi (bool, optional): check if lobes are in upper hemisphere. Defaults to False.
            fix_specular_albedo (bool, optional): _description_. Defaults to False.
            init_r_and_s (bool, optional): It will initialize roughness and specular if setting true. Defaults to False.
            init_roughness (float, optional): Initial value of roughness 0~1. Defaults to 0.7.
            init_specular (float, optional): Initial value of specular 0~1. Defaults to 0.23.
            material_nerf_pe_numfreq (int, optional): the numer of frequencies using nerf's position encoding in svbrdf network. Defaults to 0.
        """
        super().__init__(
                material_random_pe_numfreq, 
                dim,
                material_random_pe_sigma,
                white_specular,
                white_light,
                num_lgt_sgs,
                num_base_materials,
                upper_hemi,
                init_r_and_s,
                init_roughness,
                init_specular,
                material_nerf_pe_numfreq,
                num_hidden_layers,
                has_diffuse,
                has_specular,
                num_geometry_ids,
                one_net_per_component)
    
    def create_lights(self):
        print('Number of Light SG: ', self.numLgtSGs)
        print('Number of BRDF SG: ', self.numBrdfSGs)
        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization

        if self.white_light:
            print('Using white light!')
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 5), requires_grad=True)   # [M, 5]; lobe + lambda + mu -> mu [0, 1, 2] (lobe axis), lambda [3] (lobe sharpness), a [4] (lobe amplitude)
            # self.lgtSGs.data[:, -1] = torch.clamp(torch.abs(self.lgtSGs.data[:, -1]), max=0.01)
        else:
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu -> mu [0, 1, 2] (lobe axis), lambda [3] (lobe sharpness), a [4, 5, 6] (lobe amplitude)
            self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
            # self.lgtSGs.data[:, -3:] = torch.clamp(torch.abs(self.lgtSGs.data[:, -3:]), max=0.01)

        # make sure lambda is not too
        # close to zero
        self.lgtSGs.data[:, 3:4] = 20. + torch.abs(self.lgtSGs.data[:, 3:4] * 100.)
        # make sure total energy is around 1.
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        self.lgtSGs.data[:, :3] = torch.from_numpy(lobes)
        # check if lobes are in upper hemisphere
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)

            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)
    
    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs

    def forward(self, points, geometry_ids):
        ret = svbrdf_network.forward(self, points, geometry_ids)
        
        # Spherical Gaussian Lights
        lgtSGs = self.lgtSGs
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)
            
        # Quick Hack for shininess
        # roughness[:,:] = 0
        # specular_reflectacne[:,:]= 1.0
        
        ret['sg_lgtSGs'] = lgtSGs
        
        return ret
    
    def render(self, sg_envmap_material, normal, viewdirs, diffuse_rgb=None):
        lgtSGs = sg_envmap_material['sg_lgtSGs']
        specular_reflectance = sg_envmap_material['sg_specular_reflectance']
        roughness = sg_envmap_material['sg_roughness']
        diffuse_albedo = sg_envmap_material['sg_diffuse_albedo']
        
        return render_with_sg(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, diffuse_rgb, self.has_diffuse, self.has_specular)
    
class svbrdf_point_network(nn.Module):
    def __init__(self, 
                 num_point_lights = 32,
                 num_geometry_ids=1,
                 white_specular = True,
                 white_light = True,
                 has_diffuse = True,
                 has_specular = True):
        
        super().__init__()
        self.num_geometry_ids = num_geometry_ids
        self.num_point_lights = num_point_lights
        self.white_specular = white_specular
        self.white_light = white_light
        
        self.has_diffuse = has_diffuse
        self.has_specular = has_specular
        
        self.create_lights()
        self.create_phong_material()
        
    def create_lights(self):
        print('Number of Points Lights: ', self.num_point_lights)

        if self.white_light:
            print('Using white point lights!')
            # pos x y z, color rgb
            self.point_lights = nn.Parameter(torch.randn(self.num_point_lights, 4), requires_grad=True)   # [M, 4]; pos (xyz) [0, 1, 2], intensity [3]
        else:
            self.point_lights = nn.Parameter(torch.randn(self.num_point_lights, 6), requires_grad=True)   # [M, 6]; pos (xyz) [0, 1, 2], rgb color [3, 4, 5]
            # self.point_lights.data[:, -2:] = self.point_lights.data[:, -3:-2].expand((-1, 2))      # init with 50 shades of grey
        
        point_pos = fibonacci_sphere(self.num_point_lights).astype(np.float32)
        self.point_lights.data[:, :3] = torch.from_numpy(point_pos)
        self.point_lights.data[:, 3:] = 1.0 / self.num_point_lights
    
    def create_phong_material(self):
        if self.has_diffuse:
            self.diffuse_albedo = nn.Parameter(0.5 * torch.ones(self.num_geometry_ids, 3), requires_grad = True)
        
        if self.has_specular:
            self.specular_exponents = nn.Parameter(10 * torch.ones(self.num_geometry_ids, 1), requires_grad = True)
        
            num_specular_albedo_channels = 3
            if self.white_specular:
                num_specular_albedo_channels = 1
                
            self.specular_albedo = nn.Parameter(torch.ones(self.num_geometry_ids, num_specular_albedo_channels), requires_grad = True)
        
    def forward(self, points, geometry_ids):
        ret = {}
        ret['sg_diffuse_albedo'] = None
        ret['phong_specular_exponent'] = None
        ret['sg_specular_albedo'] = None
        
        if self.has_specular:
            ret['phong_specular_exponent'] = self.specular_exponents[geometry_ids]
            ret['sg_specular_albedo'] = self.specular_albedo[geometry_ids]
            
        if self.has_diffuse:
            ret['sg_diffuse_albedo'] = self.diffuse_albedo[geometry_ids]
        
            # specular_reflectance = specular_reflectance.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])
        
        point_lights = self.point_lights
        if self.white_light:
            point_lights = torch.cat((point_lights, point_lights[..., -1:], point_lights[..., -1:]), dim = -1)
        
        ret['phong_point_lights'] = point_lights
        
        ret['phong_light_vector'] = point_lights[:, :3].unsqueeze(0) - points.unsqueeze(1) # [1, 64, 3] - [23657, 1, 3] -> [23657, 64, 3] 
        light_vec_len = torch.norm(ret['phong_light_vector'], dim = -1) + TINY_NUMBER#
        
        ret['phong_light_vector'] = ret['phong_light_vector'] / light_vec_len.unsqueeze(-1).expand(-1, -1, 3)
        
        return ret
    
    def render(self, phong_envmap_material, normal, viewdirs, diffuse_rgb=None):
        point_lights = phong_envmap_material['phong_point_lights']
        diffuse_albedo = phong_envmap_material['sg_diffuse_albedo']
        specular_albedo = phong_envmap_material['sg_specular_albedo']
        specular_exponent = phong_envmap_material['phong_specular_exponent']
        light_vec = phong_envmap_material['phong_light_vector']
        
        return render_with_point_lights(point_lights, light_vec, diffuse_albedo, specular_albedo, specular_exponent, normal, viewdirs)
    
    #Convert rectangular coordinate system to spherical coordinate system
def cart2sph(cart):#cart:[batch,3] 3 include x,y,z
    hxy = torch.hypot(cart[:,0], cart[:,1])
    r = torch.hypot(hxy, cart[:,2]).view(-1,1)
    theta = torch.atan2(hxy,cart[:,2] ).view(-1,1)
    phi = torch.atan2 (cart[:,1] , cart[:,0]).view(-1,1)
    sph = torch.cat((r,theta,phi),dim=1)
    return sph

    #Convert spherical coordinate system to rectangular coordinate system
def sph2cart(sph):
    rsin_theta = sph[:,0] * torch.sin(sph[:,1])
    x = (rsin_theta * torch.cos(sph[:,2])).view(-1,1)
    y = (rsin_theta * torch.sin(sph[:,2])).view(-1,1)
    z = (sph[:,0] * torch.cos(sph[:,1])).view(-1,1)
    cart = torch.cat((x,y,z),dim=1)
    return cart

class Normal_estimation_network(nn.Module):
    def __init__(self,
                 normal_nerf_pe_numfreq=True,
                 normal_random_pe_numfreq=False,
                 max_delta_theta=np.pi/3,
                 max_delta_phi=np.pi/3,
                 normal_random_pe_sigma=20,
                 if_normal_clamp = False,
                 num_hidden_layers = 2,
                 num_geometry_ids = 1,
                 one_net_per_component = False):
        super().__init__()
        self.normal_nerf_pe_numfreq = normal_nerf_pe_numfreq
        self.normal_random_pe_numfreq = normal_random_pe_numfreq
        self.max_delta_theta=max_delta_theta
        self.max_delta_phi=max_delta_phi
        self.if_normal_clamp = if_normal_clamp
        self.activate = nn.PReLU()
        # self.activation_type = 'prelu'
        self.one_net_per_component = one_net_per_component
        
        block_type = LinearPlusActivation #ResNetBlockPlusActivation # 
        
        def make_mlp_normal():
            normal_layers=[]
            if self.normal_nerf_pe_numfreq>0:
                self.embed_fn, input_dim = get_embedder(self.normal_nerf_pe_numfreq)
                normal_layers.append(nn.Sequential(nn.Linear( input_dim+2 , 256),self.activate))
            elif self.normal_random_pe_numfreq>0:
                normal_layers.append(FourierFeatureTransform(3, self.normal_random_pe_numfreq, normal_random_pe_sigma))
                normal_layers.append(nn.Sequential(nn.Linear(self.normal_random_pe_numfreq*2+3+2, 256),self.activate))
            else:
                normal_layers.append(nn.Sequential(nn.Linear(5, 256), self.activate))

            for i in range(num_hidden_layers): # num_hidden_layers
                normal_layers.append(block_type(256, self.activate))
            normal_layers.append(nn.Linear(256, 2))
            normal_layers.append(nn.Tanh())
            return nn.ModuleList(normal_layers)
        
        if one_net_per_component:
            self.mlp_normal = torch.nn.ModuleList([make_mlp_normal() for geom_id in range(num_geometry_ids)])
        else:
            self.mlp_normal = make_mlp_normal()
        
    def forward(self, points, normals, geometry_id = None):
        normals_sph = cart2sph(normals)
        normals_sph2 = normals_sph[:,1:]
        
        if self.normal_nerf_pe_numfreq > 0:
            x = self.embed_fn(x)
        
        if self.one_net_per_component:
            geom_ids = torch.unique(geometry_id).detach().cpu().numpy()
            x = {geom_id: points[geometry_id == geom_id] for geom_id in geom_ids}
            normals_sph2 = {geom_id: normals_sph2[geometry_id == geom_id] for geom_id in geom_ids}
        else:
            x = points
            
        if self.normal_nerf_pe_numfreq>0:
            if self.one_net_per_component:
                for geom_id in geom_ids:
                    x[geom_id] = torch.cat((x[geom_id], normals_sph2[geom_id]), dim = 1)
                    for layer in self.mlp_normal[geom_id]:
                        x[geom_id] = layer(x[geom_id])
            else:
                x = torch.cat((x,normals_sph2),dim=1)
                for layer in self.mlp_normal:
                    x = layer(x)
                    
        elif self.normal_random_pe_numfreq>0:
            if self.one_net_per_component:
                for geom_id in geom_ids:
                    x[geom_id] = self.mlp_normal[geom_id][0](x[geom_id])
                    x[geom_id] = torch.cat((x[geom_id],normals_sph2[geom_id]), dim = 1)
                    for layer in self.mlp_normal[geom_id][1:]:
                        x[geom_id] = layer(x[geom_id])
            else:
                x = self.mlp_normal[0](x)
                x = torch.cat((x,normals_sph2),dim=1)
                for layer in self.mlp_normal[1:]:
                    x = layer(x)
        else:
            if self.one_net_per_component:
                for geom_id in geom_ids:
                    x[geom_id] = torch.cat((x[geom_id], normals_sph2[geom_id]), dim = 1)
                    for layer in self.mlp_normal[geom_id]:
                        x[geom_id] = layer(x[geom_id])
                        
                output = torch.zeros((points.shape[0], 2), dtype = points.dtype, device = points.device)
                for geom_id in geom_ids:
                    output[geom_id == geometry_id] = x[geom_id]
                x = output
            else:
                x = torch.cat((x,normals_sph2), dim = 1)
                for layer in self.mlp_normal:
                    x = layer(x)

        x = x * torch.tensor([self.max_delta_theta, self.max_delta_phi]).cuda()
        
        normals_sph[:,1] +=x [:,0]
        normals_sph[:,2] +=x [:,1]
        # ipdb.set_trace()
        
        if self.if_normal_clamp:
            normals_sph[:,1] = torch.clamp(normals_sph[:,1].clone(), 0, np.pi)
            normals_sph[:,2] = torch.clamp(normals_sph[:,2].clone(), 0, np.pi * 2)
        normals_disp = sph2cart(normals_sph)
        
        return normals_disp