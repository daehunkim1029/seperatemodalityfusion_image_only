import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes  # CameraInstance3DBoxes 대신 LiDARInstance3DBoxes 사용
import matplotlib.pyplot as plt

# nuScenes 데이터셋 로드
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/spalab/sdb/kypark/SeparateModalityFusion/data/nuscenes', verbose=True)

# 이미지 파일 경로
img_path = 'samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915428762465.jpg'

# sample_data_token을 찾기 위한 함수
def get_sample_data_token(nusc, img_path):
    for sample in nusc.sample:
        for channel in sample['data']:
            sample_data = nusc.get('sample_data', sample['data'][channel])
            if img_path in sample_data['filename']:
                print(f"Found matching sample_data: {sample_data['filename']}")
                return sample_data['token']
    return None

# 이미지 파일 경로에 해당하는 sample_data_token 찾기
sample_data_token = get_sample_data_token(nusc, img_path)

if sample_data_token:
    # 샘플 데이터 로드
    sample_data = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sample_data['sample_token'])
    
    sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(sensor['camera_intrinsic'], dtype=np.float32)
    
    # ego pose 로드
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    # lidar2cam 변환 행렬 생성 (3x4 행렬)
    rotation_matrix = Quaternion(sensor['rotation']).rotation_matrix
    translation = np.array(sensor['translation']).reshape(3, 1)
    lidar2cam = np.hstack([rotation_matrix, translation])
    
    # lidar2img 변환 행렬 계산 (3x4 행렬)
    lidar2img = cam_intrinsic @ lidar2cam
    
    # 3D 바운딩 박스 생성
    bboxes_3d = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        # 글로벌 좌표에서 ego 차량 좌표로 변환
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        # ego 차량 좌표계에서의 박스 정보 저장
        bbox = np.array([
            box.center[0], box.center[1], box.center[2],
            box.wlh[1], box.wlh[0], box.wlh[2],  # LiDAR 좌표계에 맞게 w와 l을 바꿈
            box.orientation.yaw_pitch_roll[0]
        ])
        bboxes_3d.append(bbox)
        print(f"Box added: center={box.center}, wlh={box.wlh}, yaw={box.orientation.yaw_pitch_roll[0]}")
    
    # 3D 바운딩 박스를 LiDARInstance3DBoxes 형식으로 변환
    if bboxes_3d:
        bboxes_3d_np = np.array(bboxes_3d, dtype=np.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(bboxes_3d_np, box_dim=7, origin=(0.5, 0.5, 0.5))
        
        # 메타데이터 설정
        input_meta = {
            'cam2img': cam_intrinsic,
            'lidar2cam': lidar2cam,  # lidar2cam 변환 추가
            'lidar2img': lidar2img
        }
        
        # Visualizer 초기화
        visualizer = Det3DLocalVisualizer()
        
        # 이미지 불러오기
        img_full_path = f'/media/spalab/sdb/kypark/SeparateModalityFusion/data/nuscenes/{img_path}'
        img = mmcv.imread(img_full_path)
        
        # 이미지 설정
        visualizer.set_image(img)
        
        # 3D 바운딩 박스를 이미지에 투영
        visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)
        
        # 이미지 얻기
        visualized_img = visualizer.get_image()
        
        # 이미지 저장
        save_path = '/media/spalab/sdb/kypark/SeparateModalityFusion/vis_result_gt/visualized_image_ego.png'
        mmcv.imwrite(visualized_img, save_path)
        print(f"Image saved at {save_path}")
    else:
        print("No valid 3D bounding boxes found in the ego vehicle view.")
else:
    print("Sample data token not found for the given image path.")