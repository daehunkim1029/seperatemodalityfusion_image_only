from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
from mmdet3d.structures.ops import box_np_ops
import cv2
@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        freeze_img=False,
        freeze_pts=False,
        sep_fg=False,
        smt=False,
        use_pts_feat=False,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        imgpts_neck: Optional[dict] = None,
        masking_encoder: Optional[dict] = None,
        img_backbone_decoder: Optional[dict] = None,
        img_neck_decoder: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if 'voxelize_cfg' in data_preprocessor:
            voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
            self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        else:
            voxelize_cfg = None
            self.voxelize_reduce = None
            self.pts_voxel_layer = None

        if 'pillarize_cfg' in data_preprocessor:
            pillarize_cfg = data_preprocessor.pop('pillarize_cfg')
        else:
            pillarize_cfg = False

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if pillarize_cfg:
            self.pts_pillar_layer = Voxelization(**pillarize_cfg)
        else:
            self.pts_pillar_layer = False

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder) if pts_voxel_encoder else None

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder) if pts_middle_encoder else None

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone) if pts_backbone else None
        self.pts_neck = MODELS.build(pts_neck) if pts_neck else None
        self.imgpts_neck = MODELS.build(imgpts_neck) if imgpts_neck else None
        self.masking_encoder = MODELS.build(masking_encoder) if masking_encoder else None
        self.head_name = bbox_head['type']
        self.bbox_head = MODELS.build(bbox_head)
        self.img_backbone_decoder = MODELS.build(img_backbone_decoder) if img_backbone_decoder else None
        self.img_neck_decoder = MODELS.build(img_neck_decoder) if img_neck_decoder else None
        self.use_pts_feat = use_pts_feat
        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        self.sep_fg = sep_fg
        self.smt = smt
        self.init_weights()
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()
        if self.freeze_img:
            if self.img_backbone is not None:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            if self.pts_voxel_encoder is not None:
                for param in self.pts_voxel_encoder.parameters():
                    param.requires_grad = False
            if self.pts_middle_encoder is not None:
                for param in self.pts_middle_encoder.parameters():
                    param.requires_grad = False
            if self.fusion_layer is not None:
                for param in self.fusion_layer.parameters():
                    param.requires_grad = False
            # for name, param in self.named_parameters():
            #     if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.decoder.0' in name:
            #         param.requires_grad = False
            #     if 'imgpts_neck.shared_conv_pts' in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.prediction_heads.0' in name:
            #         param.requires_grad = False
            #     if 'pts_bbox_head.class_encoding' in name:
            #         param.requires_grad = False
            # def fix_bn(m):
            #     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            #         m.track_running_stats = False
            # self.pts_voxel_layer.apply(fix_bn)
            # self.pts_voxel_encoder.apply(fix_bn)
            # self.pts_middle_encoder.apply(fix_bn)
            # self.pts_backbone.apply(fix_bn)
            # self.pts_neck.apply(fix_bn)
            # self.pts_bbox_head.heatmap_head.apply(fix_bn)
            # self.pts_bbox_head.class_encoding.apply(fix_bn)
            # self.pts_bbox_head.decoder[0].apply(fix_bn)
            # self.pts_bbox_head.prediction_heads[0].apply(fix_bn)            
            # self.imgpts_neck.shared_conv_pts.apply(fix_bn)
        
        
    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None
    def visualize_feat(self, bev_feat, idx):
        
        feat = bev_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        #sum_image_feature = (np.sum(np.transpose(image_features,(1,2,0)),axis=2)/64).astype("uint8")
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        #sum_image_feature = cv2.applyColorMap(sum_image_feature,cv2.COLORMAP_JET)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        #cv2.imwrite(f"max_{idx}.jpg",sum_image_feature)
        cv2.imwrite(f"max_{idx}.jpg",max_image_feature)
        
    def visualize_bev(self, point_cloud, gt_boxes, pred_boxes, feat, start_idx=0, end_idx=400):
        point_cloud = point_cloud.cpu()
        gt_boxes = gt_boxes.cpu()
        pred_boxes = pred_boxes.cpu()
        # 2D 플롯 생성
        fig, ax = plt.subplots(figsize=(8, 8))

        # 포인트 클라우드 시각화
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1, c='gray')

        # GT 바운딩 박스 시각화
        for box in gt_boxes:
            x, y, _, l, w, _, rot = box[:7].numpy()
            rot = -rot
            box_points = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
            rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            box_points = np.dot(box_points, rotation_matrix)
            box_points += np.array([x, y])
            ax.plot(box_points[:, 0], box_points[:, 1], c='green')

        # 예측 바운딩 박스 시각화 (묶음별로 다른 색상 사용)
        pred_colors = ['red', 'blue', 'orange']
        pred_labels = ['Fused Pred Boxes', 'Image Pred Boxes', 'LiDAR Pred Boxes']

        num_pred_boxes = len(pred_boxes)
        start_idx = max(0, min(start_idx, num_pred_boxes))
        end_idx = max(start_idx, min(end_idx, num_pred_boxes))
        pred_boxes = pred_boxes[start_idx:end_idx]

        group_sizes = [200, 100, 100]
        group_start_indices = [0] + np.cumsum(group_sizes).tolist()[:-1]

        for i, group_start_idx in enumerate(group_start_indices):
            group_end_idx = min(group_start_idx + group_sizes[i], end_idx)
            if group_start_idx >= group_end_idx:
                break

            boxes = pred_boxes[max(0, group_start_idx - start_idx):max(0, group_end_idx - start_idx)]
            for box in boxes:
                #y, x, _, w, l, _, rot = box[:7].numpy()
                x, y, _, l, w, _, rot = box[:7].numpy()
                #rot = -(rot+np.pi/2)
                rot = -rot
                box_points = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
                rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                box_points = np.dot(box_points, rotation_matrix)
                box_points += np.array([x, y])
                ax.plot(box_points[:, 0], box_points[:, 1], c=pred_colors[i])

        # 축 레이블 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

        # 색상에 대한 범례 표시
        for color, label in zip(pred_colors, pred_labels):
            ax.plot([], [], c=color, label=label)
        ax.legend(loc='upper right')

        # 시각화 결과 저장
        plt.tight_layout()
        plt.savefig(f"visualization_bev_color_legend_{feat}.png")
        plt.close()
        
    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        pts_feats,
        pts_metas,
        fg_bg_mask_list=None,
        sensor_list=None,
        batch_input_metas=None
    ) -> torch.Tensor:

        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()
        #self.visualize_feat(x[0],'0')
        # x_ = x.clone()
        x = self.img_backbone(x)
        x = self.img_neck(x)
        #self.visualize_feat(x[0][0],'1')
        if not isinstance(x, torch.Tensor):
            x = x[0]
        
        BN, C, H, W = x.size()
        if self.imgpts_neck is not None:
            x, pts_feats, mask_loss = self.imgpts_neck(x, pts_feats, img_metas, pts_metas, fg_bg_mask_list, sensor_list, batch_input_metas)#, img=x_, points=points)
            x = x.contiguous()
        x = x.view(B, int(BN / B), C, H, W)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        #import pdb; pdb.set_trace()
        #self.visualize_feat( x[0], '2')
        if self.imgpts_neck is not None:
            return x, pts_feats, mask_loss
        else:
            return x, pts_feats, None

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        if 'points' not in batch_inputs_dict or self.pts_voxel_layer is None:
            return None, None
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            if self.pts_pillar_layer:
                pts_metas = self.voxelize(points, voxel_type='pillar')
            else:
                pts_metas = None
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x, pts_metas

    @torch.no_grad()
    def voxelize(self, points, voxel_type='voxel'):
        if self.pts_voxel_layer is None:
            return None, None , None
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            if voxel_type == 'voxel':
                ret = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                ret = self.pts_pillar_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n) # num_points
        feats = torch.cat(feats, dim=0) # voxels
        coords = torch.cat(coords, dim=0) # coors
        
        if voxel_type == 'pillar':
            pts_metas = {}
            pts_metas['pillars'] = feats
            pts_metas['pillar_coors'] = coords
            pts_metas['pts'] = points
        
        # HardSimpleVFE
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        if voxel_type == 'pillar':
            pts_metas['pillar_center'] = feats
            pts_metas['pillars_num_points'] = sizes
            return pts_metas
        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _, _, cm_feat = self.extract_feat(batch_inputs_dict, batch_input_metas)

        #self.visualize_feat( feats[0], '4')

        if self.with_bbox_head:
            if self.head_name == "RobustHead":
                outputs = self.bbox_head.predict(feats, cm_feat, batch_input_metas)
            else:
                outputs = self.bbox_head.predict(feats, batch_data_samples,**kwargs)
        
        
        
        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        #points = batch_inputs_dict['points'][0]
        #gt = res[0].eval_ann_info['gt_bboxes_3d'].tensor
        #pred = res[0].pred_instances_3d.bboxes_3d.tensor
        # self.visualize_bev(points, gt, pred, 'no', start_idx=0, end_idx=0)
        #self.visualize_bev(points, gt, pred, '18', start_idx=0, end_idx=500)
        #import pdb; pdb.set_trace()
        # self.visualize_bev(points, gt, pred, 'img', start_idx=200, end_idx=300)
        # self.visualize_bev(points, gt, pred, 'lidar', start_idx=300, end_idx=400) 
        # lidar_path = batch_data_samples[0].metainfo['lidar_path']

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        fg_bg_mask_list=None,
        sensor_list=None,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        #import pdb; pdb.set_trace()
        pts_feature, pts_metas = self.extract_pts_feat(batch_inputs_dict)
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, pts_feature, mask_loss = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas,
                                                pts_feature,
                                                pts_metas,
                                                fg_bg_mask_list,
                                                sensor_list,
                                                batch_input_metas)
            features.append(img_feature)
        features.append(pts_feature)
        if self.fusion_layer is not None:
            if 'mask_ratio' in self.fusion_layer.__dict__:
                x, pts_loss = self.fusion_layer(features, fg_bg_mask_list, sensor_list, batch_input_metas)
            else:
                x = self.fusion_layer(features)
                pts_loss = None
        else:
            #assert len(features) == 1, features
            x = features[0]
            pts_loss = None
        #x = self.pts_backbone(x)
        #x = self.pts_neck(x)
        if self.img_backbone_decoder is not None:
            img_feature = self.img_backbone_decoder(x)
        if self.img_neck_decoder is not None:   
            img_feature = self.img_neck_decoder(img_feature)
        if self.use_pts_feat:
            pts_feature = self.pts_backbone(pts_feature.clone())
            pts_feature = self.pts_neck(pts_feature)
            pts_feature = pts_feature[0]
        #print('bev_x',img_feature.shape)
        #self.visualize_feat( img_feature[0], '3')
        #import pdb; pdb.set_trace()
        return img_feature, mask_loss, pts_loss, [img_feature, pts_feature]

    def fg_bg_mask(self, batch_data_samples):
            
        grid_size = torch.tensor([1440, 1440, 1])
        pc_range = torch.tensor([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        voxel_size = torch.tensor([0.075, 0.075, 0.2])

        B, H, W = len(batch_data_samples), 180, 180

        assert grid_size[0] == grid_size[1] and W == H
        assert grid_size[0] % W == 0
        out_size_factor = torch.div(grid_size[0], W, rounding_mode='floor')

        coord_xs = [i * voxel_size[0] * out_size_factor + pc_range[0] for i in range(W)]
        coord_ys = [i * voxel_size[1] * out_size_factor + pc_range[1] for i in range(H)]
        coord_xs, coord_ys = np.meshgrid(coord_xs, coord_ys, indexing='ij')
        coord_xs = coord_xs.reshape(-1, 1)
        coord_ys = coord_ys.reshape(-1, 1)

        coord_zs = np.ones_like(coord_xs) * 0.5
        coords = np.hstack((coord_xs, coord_ys, coord_zs))
        assert coords.shape[0] == W * W and coords.shape[1] == 3
        
        device = torch.device('cpu')
        coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        
        fg_masks = []
        bg_masks = []
        
        for sample in batch_data_samples:
            boxes = sample.gt_instances_3d['bboxes_3d']
            points = coords.numpy()
            boxes = deepcopy(boxes.detach().cpu().numpy())
            boxes[:, 2] = 0
            boxes[:, 5] = 1
            mask = box_np_ops.points_in_rbbox(points, boxes)

            fg_mask = mask.any(axis=-1).astype(float)
            bg_mask = abs(fg_mask-1)

            fg_mask = fg_mask.reshape(1, 1, H, W)
            bg_mask = bg_mask.reshape(1, 1, H, W)
            fg_masks.append(torch.tensor(fg_mask))
            bg_masks.append(torch.tensor(bg_mask))

        fg_mask = torch.cat(fg_masks, dim=0).float()
        bg_mask = torch.cat(bg_masks, dim=0).float()

        return [fg_mask, bg_mask]

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.sep_fg:
            fg_bg_mask_list = self.fg_bg_mask(batch_data_samples)
        else:
            fg_bg_mask_list = None
        if 'sensor_list' in batch_inputs_dict:
            sensor_list = batch_inputs_dict['sensor_list']
        else:
            sensor_list = None
        feats, mask_loss, pts_loss, cm_feat = self.extract_feat(batch_inputs_dict, batch_input_metas, fg_bg_mask_list, sensor_list)
        losses = dict()
        if self.with_bbox_head:
            if self.head_name == 'RobustHead':
                bbox_loss = self.bbox_head.loss(feats, cm_feat, batch_data_samples)
            else:
                bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        if pts_loss:
            if isinstance(pts_loss,dict):
                losses.update(pts_loss) 
        if mask_loss:
            if isinstance(mask_loss,dict):
                losses.update(mask_loss) 
        losses.update(bbox_loss)
        return losses