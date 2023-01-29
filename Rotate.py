import torch
import torch.nn.functional as F
def get_3d_locations(d, h, w, device_):
    locations_x = torch.linspace(0, w - 1, w).view(1, 1, 1, w).to(device_).expand(1, d, h, w)
    locations_y = torch.linspace(0, h - 1, h).view(1, 1, h, 1).to(device_).expand(1, d, h, w)
    locations_z = torch.linspace(0, d - 1, d).view(1, d, 1, 1).to(device_).expand(1, d, h, w)
    # stack locations
    locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
    return locations_3d


def rotate(input_tensor, rotation_matrix):
    device_ = input_tensor.device
    _, d, h, w = input_tensor.shape
    input_tensor = input_tensor.unsqueeze(0)
    # get x,y,z indices of target 3d data
    locations_3d = get_3d_locations(d, h, w, device_)
    # rotate target positions to the source coordinate
    rotated_3d_positions = torch.bmm(rotation_matrix.view(1, 3, 3).expand(d * h * w, 3, 3), locations_3d).view(1, d, h,
                                                                                                               w, 3)
    rot_locs = torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4)
    # change the range of x,y,z locations to [-1,1]
    normalised_locs_x = (2.0 * rot_locs[0] - (w - 1)) / (w - 1)
    normalised_locs_y = (2.0 * rot_locs[1] - (h - 1)) / (h - 1)
    normalised_locs_z = (2.0 * rot_locs[2] - (d - 1)) / (d - 1)
    grid = torch.stack([normalised_locs_x, normalised_locs_y, normalised_locs_z], dim=4).view(1, d, h, w, 3)
    # here we use the destination voxel-positions and sample the input 3d data trilinearly
    rotated_signal = F.grid_sample(input=input_tensor, grid=grid, mode='nearest', align_corners=True)
    return rotated_signal.squeeze(0)