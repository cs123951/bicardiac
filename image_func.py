import torch


def create_batch_regular_grid(batch_size, img_size, device='cpu'):
    """

    :param batch_size:
    :param img_size:
    :param device:
    :return: channel last regular grid [batch, x,y,z,3]
    """
    dim = len(img_size)
    if dim == 3:
        D, H, W = img_size
        x_range = torch.tensor([i * 2 / (D - 1) - 1 for i in range(D)], device=device)
        y_range = torch.tensor([i * 2 / (H - 1) - 1 for i in range(H)], device=device)
        z_range = torch.tensor([i * 2 / (W - 1) - 1 for i in range(W)], device=device)

        regular_grid_list = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        regular_grid = torch.stack(regular_grid_list, dim=-1)
        batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1, 1)

    elif dim == 2:
        D, H = img_size
        x_range = torch.tensor([i * 2 / (D - 1) - 1 for i in range(D)], device=device)
        y_range = torch.tensor([i * 2 / (H - 1) - 1 for i in range(H)], device=device)

        regular_grid_list = torch.meshgrid(x_range, y_range, indexing='ij')
        regular_grid = torch.stack(regular_grid_list, dim=-1)
        batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1)

    return batch_regular_grid
