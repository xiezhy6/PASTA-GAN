import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
import torch
import random, numbers, math

import torch.nn.functional as F


CMAP = sio.loadmat('./human_colormap.mat')['colormap']
CMAP = (CMAP * 256).astype(np.uint8)

#########################################
##     For visualization
#########################################
label_colors = [(0,0,0), (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0),
                (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128),
                (0,128,0), (0,0,255), (51,170,221), (0,255,255), (85,255,170),
                (170,255,85), (255,255,0), (255,170,0)]


def visualize_img(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    num = len(imgs)
    row = int(math.sqrt(num))
    col = (num + row - 1) // row
    for i, img in enumerate(imgs):
        plt.subplot(row, col, i+1)
        img = img[0].detach().cpu().numpy()
        if len(img.shape) > 2:                      # three channels
            if img.shape[0] < img.shape[2]:         # HWC -> CHW
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 3:                   # RGB image
                img = img * 0.5 + 0.5
            else:
                img = img[:, :, 0]                  # gray scale
        plt.imshow(img)
    #plt.show()
    plt.savefig('./visualization.jpg')

def visualize_joint(joints):
    if not isinstance(joints, list):
        joints = [joints]
    num = len(joints)
    for i, joint in enumerate(joints):
        plt.subplot(1, num, i+1)
        joint = joint[0].detach().cpu().numpy()
        plt.imshow(np.sum(joint, axis=0))
    plt.show()

def vis_numpy(img_ls, channel_wise=False,name='img.png'):
    if isinstance(img_ls, np.ndarray) and channel_wise:
        split_num = img_ls.shape[-1] // 3
        img_ls = np.array_split(img_ls, split_num, axis=2)
    if isinstance(img_ls, np.ndarray) and len(img_ls.shape) == 4:
        np.array_split(img_ls, 1, axis=0)
    if not isinstance(img_ls, list):
        img_ls = [img_ls]
    img_num = len(img_ls)
    h, w = int(math.sqrt(img_num)), img_num // int(math.sqrt(img_num))
    for i in range(h):
        for j in range(w):
            plt.subplot(h, w, i * w + j + 1)
            plt.imshow(img_ls[i * w + j])
    # plt.show()
    plt.savefig(name)

def make_parts_shape(parts):
    if len(parts) == 21: parts = parts[:-1 , : , :]
    shape_labels = np.argmax(parts, axis=0)
    shape = np.zeros(parts[0].shape + (3,))
    for label in range(len(label_colors)):
        shape[shape_labels==label] = label_colors[label]
    return np.array(shape/255.)


def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


#########################################
##     For contrastive learning
#########################################

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)

def flip_cihp_batch(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[:,xx,:,:].unsqueeze(1)
    tail_list_rev[14] = tail_list[:,15,:,:].unsqueeze(1)
    tail_list_rev[15] = tail_list[:,14,:,:].unsqueeze(1)
    tail_list_rev[16] = tail_list[:,17,:,:].unsqueeze(1)
    tail_list_rev[17] = tail_list[:,16,:,:].unsqueeze(1)
    tail_list_rev[18] = tail_list[:,19,:,:].unsqueeze(1)
    tail_list_rev[19] = tail_list[:,18,:,:].unsqueeze(1)
    return torch.cat(tail_list_rev,dim=1)


def parsing2im(self, parsing):
    size = parsing.size()
    for ii in range(size[0]):
        parsing_numpy = parsing[ii].cpu().float().numpy()
        image_index = parsing_numpy.astype(np.int32)

        parsing_numpy = np.zeros((image_index.shape[0], image_index.shape[1], 3))
        for h in range(image_index.shape[0]):
            for w in range(image_index.shape[1]):
                parsing_numpy[h, w, :] = CMAP[image_index[h, w]]
        parsing_numpy = np.transpose(parsing_numpy,(2,0,1))
        if ii == 0:
            parsing_tensor = parsing_numpy[np.newaxis,...]
        else:
            parsing_tensor = np.concatenate((parsing_tensor, parsing_numpy[np.newaxis,...]),axis=0)
    parsing_tensor = torch.tensor(parsing_tensor)
    parsing_tensor = parsing_tensor / 255. * 2. - 1.

    return parsing_tensor

def label2onehot(parsing_tensor):
    parsing_tensor = torch.unsqueeze(parsing_tensor,1)
    size = parsing_tensor.size()
    oneHot_size = (size[0], 20, size[2], size[3])
    parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    parsing_label = parsing_label.scatter_(1, parsing_tensor.data.long().cuda(), 1.0)

    return parsing_label


def random_affine_matrix(output_shape):
    # angle, shift, scale, shear = getRandomAffineParam(output_size=output_shape[2:])
    # center = (output_shape[3] * 0.5 + 0.5, output_shape[2] * 0.5 + 0.5)
    # affm = get_affine_matrix(center, angle, shift, scale, shear).astype(np.float32)
    affm = []
    for i in range(output_shape[0]):
        angle = (random.randint(0, 20) - 10) * math.pi / 180
        transx = np.random.rand() * 0.4 - 0.2
        transy = np.random.rand() * 0.4 - 0.2
        scalex = (np.random.rand() * 0.2 - 0.1) + 1.
        scaley = (np.random.rand() * 0.2 - 0.1) + 1.
        affm.append(np.array([[scalex * math.cos(angle), math.sin(-angle), transx], 
                        [math.sin(angle), scaley * math.cos(angle), transy]], dtype=np.float32))
    affm = np.stack(affm)
    output = torch.from_numpy(affm)     # .repeat(output_shape[0], 1, 1)
    output.requires_grad = False        # dosen't need gradiant
    return output


def getRandomAffineParam(output_size):
    # configurations
    degrees = 10
    translate = (0.1, 0.1)
    scale_ranges = (0.8, 1.2)
    shears = 0
    angle = random.uniform(-degrees, degrees)
    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if isinstance(shears, (tuple, list)):
        if len(shears) == 2:
            shears = [random.uniform(shears[0], shears[1]), 0.]
        elif len(shears) == 4:
            shears = [random.uniform(shears[0], shears[1]),
                        random.uniform(shears[2], shears[3])]
        else:
            raise ValueError("shear matrix shape error.")

    if translate is not None:
        max_dx = translate[0] * output_size[0]
        max_dy = translate[1] * output_size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
    else:
        translations = (0, 0)
    return angle, translations, scale, shears

def get_inverse_affine_matrix(center, angle, translate, scale, shear):
    if isinstance(shear, numbers.Number):
        shear = [shear, 0]

    if not isinstance(shear, (tuple, list)) and len(shear) == 2:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    M = [d, -b, 0,
            -c, a, 0]
    M = [x / scale for x in M]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
    M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    M[2] += cx
    M[5] += cy
    return M

def get_affine_matrix(center, angle, translate, scale, shear):
    matrix_inv = get_inverse_affine_matrix(center, angle, translate, scale, shear)
    matrix_inv = np.matrix(matrix_inv).reshape(2,3)
    pad = np.matrix([0,0,1])
    matrix_inv = np.concatenate((matrix_inv, pad), 0)
    matrix = np.linalg.inv(matrix_inv)
    return matrix


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
            (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


########## for patch discriminator ########
def apply_random_crop(x, valid_coordinates, target_size, scale_range, num_crops=1):
    # build grid
    bz = x.size(0)
    crop_list = []
    for ii in range(bz):
        coordinate_ii = valid_coordinates[ii]
        x_ii = x[ii:ii+1,:,coordinate_ii[2]:coordinate_ii[3]+1,coordinate_ii[0]:coordinate_ii[1]+1]
        
        # from util_functions import visualize_img
        # visualize_img(x_ii,name=str(ii)+'.jpg')
        
        flip = torch.round(torch.rand(num_crops,1,1,1,device=x.device)) * 2 - 1.0
        unit_grid_x = torch.linspace(-1.0,1.0,target_size,device=x.device)[np.newaxis,np.newaxis,:,np.newaxis].repeat(num_crops,target_size,1,1)
        unit_grid_y = unit_grid_x.transpose(1,2)
        unit_grid = torch.cat([unit_grid_x*flip,unit_grid_y],dim=3)

        x_ii = x_ii.expand(num_crops,-1,-1,-1)
        scale = torch.rand(num_crops, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        offset = (torch.rand(num_crops, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
        sampling_grid = unit_grid * scale + offset

        # if x_ii.size(2) == 0:
        #     print('--------------------')
        #     print(coordinate_ii)
        #     print(x.size())
        #     print(x_ii.size())
        #     print('------------------')
        #     from util_functions import visualize_img
        #     visualize_img(x_ii,'wrong.jpg')

        crop = F.grid_sample(x_ii, sampling_grid, align_corners=False)
        crop_list.append(crop.unsqueeze(0))

    crop = torch.cat(crop_list,dim=0)
    return crop

def get_random_crops(x, valid_coordinates, patch_size, patch_min_scale, patch_max_scale, patch_num_crops):
    """ Make random crops.
        Corresponds to the yellow and blue random crops of Figure 2.
    """
    crops = apply_random_crop(
        x, valid_coordinates, patch_size,
        (patch_min_scale, patch_max_scale),
        num_crops=patch_num_crops
    )
    return crops
