#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers import Conv, conv_disp, RNN_block, group_operation, cat_seq_along_channel
import image_func


class GroupwiseModel(nn.Module):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 n_steps=3,
                 depthwise=False):
        super(GroupwiseModel, self).__init__()
        self.dim = len(img_size)  # vol_size is [272, 240, 96], so dim is 3
        self.device_ids = device_ids
        self.input_device = input_device
        self.output_device = output_device
        self.dim = len(img_size)
        self.img_size = img_size
        self.n_steps = n_steps
        self.depthwise = depthwise

    def loss_by_input(self, img_seq, loss_function=None):
        disp_tb3xyz = self.forward(img_seq)
        loss, loss_part = self.loss(loss_function, disp_tb3xyz)
        return loss, loss_part

    def loss(self, loss_function, disp_tb3xyz):
        if loss_function is None:
            return torch.tensor(
                0.0, device=self.output_device), [torch.tensor(0.0, device=self.output_device)]
        kl_loss = torch.tensor(0.0, device=self.output_device)
        if self.conv_type == 'BBBconv':
            count = 0
            kl_t = torch.tensor(0.0, device=self.output_device)
            for module in self.modules():
                if hasattr(module, 'kl_loss'):
                    kl_t += module.kl_loss()
                    count += 1
            kl_loss += kl_t / count

        self.img_tb1xyz = [
            self.img_tb1xyz[i].cuda(self.output_device) for i in range(len(self.img_tb1xyz))
        ]
        loss, loss_part = loss_function(disp_tb3xyz, self.batch_regular_grid, self.img_tb1xyz,
                                        kl_loss)
        return loss, loss_part


# formal CRNet
class CRNet(GroupwiseModel):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 down_out_channel_list,
                 num_layers,
                 hidden_dim_list,
                 kernel_size_list,
                 n_steps=0,
                 depthwise=False,
                 batch_first=True,
                 bias=True,
                 rnn_cell='ConvGRUCell',
                 cell_params=dict(),
                 dropout=-1,
                 conv_type='conv',
                 add_pair=False):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps, depthwise)

        self.rnn_device = output_device
        self.disp_device = output_device  # self.device_ids[0]

        self.scale_factor = 4
        self.interp_mode = "trilinear" if self.dim == 3 else "bicubic"
        RNN_size = [img_size[i] // self.scale_factor for i in range(self.dim)]
        self.conv_type = conv_type
        self.add_pair = add_pair

        self.cell_params = {}
        if rnn_cell == 'ConvLSTMCell':
            self.cell_params = cell_params

        if self.depthwise:
            RNN_size[2] = img_size[2]
            self.scale_factor = [self.scale_factor, self.scale_factor, 1]
        else:
            self.scale_factor = [self.scale_factor] * self.dim

        self.enc_features = nn.Sequential(
            Conv(1, down_out_channel_list[0], down=True, dim=self.dim, depthwise=self.depthwise),
            Conv(down_out_channel_list[0],
                 down_out_channel_list[1],
                 down=True,
                 dim=self.dim,
                 depthwise=self.depthwise)).cuda(self.input_device)

        self.multiRNN = RNN_block(img_size=RNN_size,
                                  input_dim=down_out_channel_list[-1],
                                  hidden_dim_list=hidden_dim_list,
                                  kernel_size_list=kernel_size_list,
                                  num_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  batch_first=batch_first,
                                  bias=bias,
                                  dropout=dropout,
                                  conv_type=conv_type,
                                  cell_params=self.cell_params).cuda(self.rnn_device)

        if add_pair:
            self.enc_pair = nn.Sequential(
                Conv(2, down_out_channel_list[0], down=True, dim=self.dim,
                     depthwise=self.depthwise),
                Conv(down_out_channel_list[0],
                     down_out_channel_list[1],
                     down=True,
                     dim=self.dim,
                     depthwise=self.depthwise)).cuda(self.input_device)
            self.outconv3 = nn.Sequential(
                Conv(down_out_channel_list[1] * 2,
                     down_out_channel_list[0],
                     up=False,
                     dim=self.dim,
                     depthwise=False),
                conv_disp(down_out_channel_list[0], kernel_size=3, dim=self.dim),
                nn.Upsample(scale_factor=tuple(self.scale_factor),
                            align_corners=True,
                            mode='trilinear')).cuda(self.disp_device)
        else:
            self.outconv3 = nn.Sequential(
                conv_disp(down_out_channel_list[1], kernel_size=3, dim=self.dim),
                nn.Upsample(scale_factor=tuple(self.scale_factor),
                            align_corners=True,
                            mode='trilinear')).cuda(self.disp_device)
        self.batch_regular_grid = image_func.create_batch_regular_grid(1,
                                                                       img_size,
                                                                       device=self.output_device)

    def forward(self, img_seq, loss_function=None):
        """
        img_seq: seq * [batch, channel, x, y, z]
        """
        self.img_tb1xyz = img_seq = [
            img_seq[i].cuda(self.input_device) for i in range(len(img_seq))
        ]
        features = group_operation(img_seq, self.enc_features)
        # features: list of t * [batch_t, channel, x//4, y//4, z//4]

        features = [f.cuda(self.rnn_device) for f in features]
        features = self.multiRNN(features)  # t*[batch_t, channel, x, y, z]

        if self.add_pair:
            img_0 = img_seq[0]
            img_pair_seq = []
            for tdx in range(len(img_seq)):
                cur_bs = len(img_seq[tdx])
                cat_pair_t = torch.cat([img_0[:cur_bs], img_seq[tdx]], dim=1)
                img_pair_seq.append(cat_pair_t)
            pair_features = group_operation(img_pair_seq,
                                            self.enc_pair)  # t*[batch_t, out_channel, x, y, z]
            pair_features = [f.cuda(self.rnn_device) for f in pair_features]
            features = cat_seq_along_channel(pair_features, features)

        disp_tb3xyz = group_operation(features, self.outconv3)[1:]
        # print("disp_tb3xyz ", len(disp_tb3xyz), disp_tb3xyz[0].shape)
        if self.n_steps > 0:
            params_dict = {"grid": self.batch_regular_grid, "n_steps": self.n_steps}
            disp_tb3xyz = group_operation(disp_tb3xyz, image_func.integrate_displacement,
                                          **params_dict)

        loss, loss_part = self.loss(loss_function, disp_tb3xyz)
        # (seq_len-1)*[batch, 3, x, y, z]
        return disp_tb3xyz, loss, loss_part


class BiCRNet(GroupwiseModel):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 down_out_channel_list,
                 num_layers,
                 hidden_dim_list,
                 kernel_size_list,
                 n_steps=0,
                 depthwise=False,
                 batch_first=True,
                 bias=True,
                 rnn_cell='ConvGRUCell',
                 cell_params=dict(),
                 dropout=-1,
                 conv_type='conv',
                 add_pair=False):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps, depthwise)

        self.rnn_device = output_device
        self.disp_device = output_device  # self.device_ids[0]

        self.scale_factor = 4
        self.interp_mode = "trilinear" if self.dim == 3 else "bicubic"
        RNN_size = [img_size[i] // self.scale_factor for i in range(self.dim)]
        self.conv_type = conv_type
        self.add_pair = add_pair

        self.cell_params = {}
        if rnn_cell == 'ConvLSTMCell':
            self.cell_params = cell_params

        if self.depthwise:
            RNN_size[2] = img_size[2]
            self.scale_factor = [self.scale_factor, self.scale_factor, 1]
            self.up_scale = [2, 2, 1]
        else:
            self.scale_factor = [self.scale_factor] * self.dim
            self.up_scale = [2] * self.dim

        self.enc_features = nn.Sequential(
            Conv(1, down_out_channel_list[0], down=True, dim=self.dim, depthwise=self.depthwise),
            Conv(down_out_channel_list[0],
                 down_out_channel_list[1],
                 down=True,
                 dim=self.dim,
                 depthwise=self.depthwise)).cuda(self.input_device)

        self.forward_rnn = RNN_block(img_size=RNN_size,
                                     input_dim=down_out_channel_list[-1],
                                     hidden_dim_list=hidden_dim_list,
                                     kernel_size_list=kernel_size_list,
                                     num_layers=num_layers,
                                     rnn_cell=rnn_cell,
                                     batch_first=batch_first,
                                     bias=bias,
                                     dropout=dropout,
                                     conv_type=conv_type,
                                     cell_params=self.cell_params).cuda(self.rnn_device)
        self.backward_rnn = RNN_block(img_size=RNN_size,
                                      input_dim=down_out_channel_list[-1],
                                      hidden_dim_list=hidden_dim_list,
                                      kernel_size_list=kernel_size_list,
                                      num_layers=num_layers,
                                      rnn_cell=rnn_cell,
                                      batch_first=batch_first,
                                      bias=bias,
                                      dropout=dropout,
                                      conv_type=conv_type,
                                      cell_params=self.cell_params).cuda(self.rnn_device)
        self.rnn_conv = Conv(down_out_channel_list[1] * 2,
                             down_out_channel_list[1],
                             down=False,
                             dim=self.dim,
                             depthwise=False).cuda(self.disp_device)

        self.outconv3 = nn.Sequential(
            # Conv(down_out_channel_list[1], down_out_channel_list[1], up=True, dim=self.dim, depthwise=self.depthwise),
            # Conv(down_out_channel_list[1], down_out_channel_list[0], up=True, dim=self.dim, depthwise=self.depthwise),
            conv_disp(down_out_channel_list[1], kernel_size=3, dim=self.dim),
            nn.Upsample(scale_factor=tuple(self.scale_factor), align_corners=True,
                        mode='trilinear')).cuda(self.disp_device)

        self.batch_regular_grid = image_func.create_batch_regular_grid(1,
                                                                       img_size,
                                                                       device=self.output_device)

    def forward(self, img_seq, loss_function=None):
        """
        img_seq: seq * [batch, channel, x, y, z]
        batch can only be one
        """
        img_seq = [img_seq[i].cuda(self.input_device) for i in range(len(img_seq))]

        self.img_tb1xyz = img_seq[:]
        img_seq.append(img_seq[0])

        batch_img_seq = torch.cat(img_seq, dim=0)  # batch size can only be one
        enc_feat2 = self.enc_features(batch_img_seq)  # output channel is down_ch_list[0]

        # ===== enc_feat2 --> rnn_feat
        rnn_feat = []
        for i in range(len(enc_feat2)):
            rnn_feat.append(enc_feat2[i].unsqueeze(0).cuda(self.rnn_device))
        for_feat = self.forward_rnn(rnn_feat)  # t*[batch_t, channel, x, y, z]
        back_feat = self.backward_rnn(rnn_feat[::-1])[::-1]  # t*[batch_t, channel, x, y, z]

        for_feat = for_feat[1:-1]
        back_feat = back_feat[1:-1]

        for_feat = torch.cat(for_feat, dim=0)
        back_feat = torch.cat(back_feat, dim=0)

        rnn_feat = torch.cat([for_feat, back_feat], dim=1)  # channel is down_ch_list[1]*2
        rnn_feat = self.rnn_conv(rnn_feat)

        full_disp = self.outconv3(rnn_feat)
        disp_tb3xyz = torch.chunk(full_disp, chunks=len(full_disp), dim=0)

        if self.n_steps > 0:
            params_dict = {"grid": self.batch_regular_grid, "n_steps": self.n_steps}
            disp_tb3xyz = group_operation(disp_tb3xyz, image_func.integrate_displacement,
                                          **params_dict)

        loss, loss_part = self.loss(loss_function, disp_tb3xyz)
        # (seq_len-1)*[batch, 3, x, y, z]
        return disp_tb3xyz, loss, loss_part


class PairCRNet(GroupwiseModel):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 down_out_channel_list,
                 num_layers,
                 hidden_dim_list,
                 kernel_size_list,
                 n_steps=0,
                 depthwise=False,
                 batch_first=True,
                 bias=True,
                 rnn_cell='ConvGRUCell',
                 cell_params=dict(),
                 dropout=-1,
                 conv_type='conv',
                 add_pair=False):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps, depthwise)

        self.rnn_device = output_device
        self.disp_device = output_device  # self.device_ids[0]

        self.scale_factor = 4
        self.interp_mode = "trilinear" if self.dim == 3 else "bicubic"
        RNN_size = [img_size[i] // self.scale_factor for i in range(self.dim)]
        self.conv_type = conv_type
        self.add_pair = add_pair

        self.cell_params = {}
        if rnn_cell == 'ConvLSTMCell':
            self.cell_params = cell_params

        if self.depthwise:
            RNN_size[2] = img_size[2]
            self.scale_factor = [self.scale_factor, self.scale_factor, 1]
            self.up_scale = [2, 2, 1]
        else:
            self.scale_factor = [self.scale_factor] * self.dim
            self.up_scale = [2] * self.dim

        self.enc_features = nn.Sequential(
            Conv(2, down_out_channel_list[0], down=True, dim=self.dim, depthwise=self.depthwise),
            Conv(down_out_channel_list[0],
                 down_out_channel_list[1],
                 down=True,
                 dim=self.dim,
                 depthwise=self.depthwise)).cuda(self.input_device)

        self.rnn = RNN_block(img_size=RNN_size,
                             num_layers=num_layers,
                             rnn_cell=rnn_cell,
                             input_dim=down_out_channel_list[1],
                             hidden_dim_list=hidden_dim_list,
                             kernel_size_list=kernel_size_list,
                             batch_first=batch_first,
                             bias=bias,
                             dropout=dropout,
                             conv_type=conv_type,
                             cell_params=self.cell_params).cuda(self.rnn_device)

        self.outconv3 = nn.Sequential(
            # Conv(down_out_channel_list[1], down_out_channel_list[1], up=True, dim=self.dim, depthwise=self.depthwise),
            # Conv(down_out_channel_list[1], down_out_channel_list[0], up=True, dim=self.dim, depthwise=self.depthwise),
            conv_disp(down_out_channel_list[1], kernel_size=3, dim=self.dim),
            nn.Upsample(scale_factor=tuple(self.scale_factor), align_corners=True,
                        mode='trilinear')).cuda(self.disp_device)

        self.batch_regular_grid = image_func.create_batch_regular_grid(1,
                                                                       img_size,
                                                                       device=self.output_device)

    def forward(self, img_seq, loss_function=None):
        """
        img_seq: seq * [batch, channel, x, y, z]
        batch can only be one
        """
        self.img_tb1xyz = img_seq[:]
        img_seq_pair = [
            torch.cat([img_seq[0], img_seq[i]], dim=1).cuda(self.input_device)
            for i in range(1, len(img_seq))
        ]

        batch_img_seq_pair = torch.cat(img_seq_pair, dim=0)  # batch size can only be one
        enc_feat2 = self.enc_features(batch_img_seq_pair)  # output channel is down_ch_list[0]

        # ===== enc_feat2 --> rnn_feat
        rnn_feat = []
        for i in range(len(enc_feat2)):
            rnn_feat.append(enc_feat2[i].unsqueeze(0).cuda(self.rnn_device))
        rnn_feat = self.rnn(rnn_feat)  # t*[batch_t, channel, x, y, z]
        rnn_feat = torch.cat(rnn_feat, dim=0)  # because batch=1

        full_disp = self.outconv3(rnn_feat)
        disp_tb3xyz = torch.chunk(full_disp, chunks=len(full_disp), dim=0)

        if self.n_steps > 0:
            params_dict = {"grid": self.batch_regular_grid, "n_steps": self.n_steps}
            disp_tb3xyz = group_operation(disp_tb3xyz, image_func.integrate_displacement,
                                          **params_dict)

        loss, loss_part = self.loss(loss_function, disp_tb3xyz)
        # (seq_len-1)*[batch, 3, x, y, z]
        return disp_tb3xyz, loss, loss_part


class PairBiCRNet(GroupwiseModel):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 down_out_channel_list,
                 num_layers,
                 hidden_dim_list,
                 kernel_size_list,
                 n_steps=0,
                 depthwise=False,
                 batch_first=True,
                 bias=True,
                 rnn_cell='ConvGRUCell',
                 cell_params=dict(),
                 dropout=-1,
                 conv_type='conv',
                 add_pair=False):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps, depthwise)

        self.rnn_device = output_device
        self.disp_device = output_device  # self.device_ids[0]

        self.scale_factor = 4
        self.interp_mode = "trilinear" if self.dim == 3 else "bicubic"
        RNN_size = [img_size[i] // self.scale_factor for i in range(self.dim)]
        self.conv_type = conv_type
        self.add_pair = add_pair

        self.cell_params = {}
        if rnn_cell == 'ConvLSTMCell':
            self.cell_params = cell_params

        if self.depthwise:
            RNN_size[2] = img_size[2]
            self.scale_factor = [self.scale_factor, self.scale_factor, 1]
            self.up_scale = [2, 2, 1]
        else:
            self.scale_factor = [self.scale_factor] * self.dim
            self.up_scale = [2] * self.dim

        self.enc_features = nn.Sequential(
            Conv(2, down_out_channel_list[0], down=True, dim=self.dim, depthwise=self.depthwise),
            Conv(down_out_channel_list[0],
                 down_out_channel_list[1],
                 down=True,
                 dim=self.dim,
                 depthwise=self.depthwise)).cuda(self.input_device)

        self.forward_rnn = RNN_block(img_size=RNN_size,
                                     cell_params=self.cell_params,
                                     hidden_dim_list=hidden_dim_list,
                                     kernel_size_list=kernel_size_list,
                                     num_layers=num_layers,
                                     rnn_cell=rnn_cell,
                                     batch_first=batch_first,
                                     bias=bias,
                                     dropout=dropout,
                                     conv_type=conv_type,
                                     input_dim=down_out_channel_list[-1]).cuda(self.rnn_device)
        self.backward_rnn = RNN_block(img_size=RNN_size,
                                      rnn_cell=rnn_cell,
                                      input_dim=down_out_channel_list[-1],
                                      hidden_dim_list=hidden_dim_list,
                                      kernel_size_list=kernel_size_list,
                                      num_layers=num_layers,
                                      batch_first=batch_first,
                                      bias=bias,
                                      dropout=dropout,
                                      cell_params=self.cell_params,
                                      conv_type=conv_type).cuda(self.rnn_device)
        self.rnn_conv = Conv(down_out_channel_list[1] * 2,
                             down_out_channel_list[1],
                             down=False,
                             dim=self.dim,
                             depthwise=False).cuda(self.disp_device)

        self.outconv3 = nn.Sequential(
            conv_disp(down_out_channel_list[1], kernel_size=3, dim=self.dim),
            nn.Upsample(scale_factor=tuple(self.scale_factor), align_corners=True,
                        mode='trilinear')).cuda(self.disp_device)

        self.batch_regular_grid = image_func.create_batch_regular_grid(1,
                                                                       img_size,
                                                                       device=self.output_device)

    def forward(self, img_seq, loss_function=None):
        """
        img_seq: seq * [batch, channel, x, y, z]
        batch can only be one
        """
        self.img_tb1xyz = img_seq[:]
        img_seq_pair = [
            torch.cat([img_seq[0], img_seq[i]], dim=1).cuda(self.input_device)
            for i in range(1, len(img_seq))
        ]

        batch_img_seq_pair = torch.cat(img_seq_pair, dim=0)  # batch size can only be one
        enc_feat2 = self.enc_features(batch_img_seq_pair)  # output channel is down_ch_list[0]

        # ===== enc_feat2 --> rnn_feat
        rnn_feat = []
        for i in range(len(enc_feat2)):
            rnn_feat.append(enc_feat2[i].unsqueeze(0).cuda(self.rnn_device))
        for_feat = self.forward_rnn(rnn_feat)  # t*[batch_t, channel, x, y, z]
        back_feat = self.backward_rnn(rnn_feat[::-1])[::-1]  # t*[batch_t, channel, x, y, z]
        for_feat = torch.cat(for_feat, dim=0)
        back_feat = torch.cat(back_feat, dim=0)

        rnn_feat = torch.cat([for_feat, back_feat], dim=1)  # channel is down_ch_list[1]*2
        rnn_feat = self.rnn_conv(rnn_feat)

        full_disp = self.outconv3(rnn_feat)
        disp_tb3xyz = torch.chunk(full_disp, chunks=len(full_disp), dim=0)

        if self.n_steps > 0:
            params_dict = {"grid": self.batch_regular_grid, "n_steps": self.n_steps}
            disp_tb3xyz = group_operation(disp_tb3xyz, image_func.integrate_displacement,
                                          **params_dict)

        loss, loss_part = self.loss(loss_function, disp_tb3xyz)
        # (seq_len-1)*[batch, 3, x, y, z]
        return disp_tb3xyz, loss, loss_part
