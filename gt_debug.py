import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NuScenes 데이터셋 로드
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)

# 이미지 파일 경로
img_path = 'samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915428762465.jpg'

# 이전에 정의한 함수들 (get_sample_data_token, draw_box, plot_coordinate_system)은 그대로 사용
# sample_data_token을 찾기 위한 함수
def get_sample_data_token(nusc, img_path):
    for sample in nusc.sample:
        for channel in sample['data']:
            sample_data = nusc.get('sample_data', sample['data'][channel])
            if img_path in sample_data['filename']:
                return sample_data['token']
    return None

# 3D 바운딩 박스 그리기 함수
def draw_box(ax, vertices, edges, color='b'):
    for edge in edges:
        ax.plot3D(*zip(*[vertices[edge[0]], vertices[edge[1]]]), color=color)

# 좌표계 축 그리기 함수
def plot_coordinate_system(ax, origin, rotation, scale=1):
    axes = np.eye(3) * scale
    for i in range(3):
        axis = rotation.rotate(axes[i])
        ax.quiver(origin[0], origin[1], origin[2], 
                  axis[0], axis[1], axis[2],
                  color=['r', 'g', 'b'][i])
# 이미지 파일 경로에 해당하는 sample_data_token 찾기
sample_data_token = get_sample_data_token(nusc, img_path)

if sample_data_token:
    # 샘플 데이터 로드
    sample_data = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sample_data['sample_token'])
    
    # 센서 캘리브레이션 데이터 로드
    sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    # ego pose 로드
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    # 그래프 설정
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    titles = ['Global Coordinates', 'Ego Vehicle Coordinates', 'Camera Coordinates']
    
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # 3D 바운딩 박스 처리
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # 글로벌 좌표
        box_global = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        vertices_global = box_global.corners().T
        
        # 에고 차량 좌표
        box_ego = box_global.copy()
        box_ego.translate(-np.array(ego_pose['translation']))
        box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        vertices_ego = box_ego.corners().T
        
        # 카메라 좌표
        box_cam = box_ego.copy()
        box_cam.translate(-np.array(sensor['translation']))
        box_cam.rotate(Quaternion(sensor['rotation']).inverse)
        vertices_cam = box_cam.corners().T
        
        # 바운딩 박스 모서리 정의
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        
        # 각 좌표계에서 바운딩 박스 그리기
        draw_box(ax1, vertices_global, edges, 'b')
        draw_box(ax2, vertices_ego, edges, 'g')
        draw_box(ax3, vertices_cam, edges, 'r')
    
    # 좌표계 축 그리기
    plot_coordinate_system(ax1, [0, 0, 0], Quaternion())
    plot_coordinate_system(ax2, [0, 0, 0], Quaternion())
    plot_coordinate_system(ax3, [0, 0, 0], Quaternion())
    
    # 축 범위 설정
    for ax in axes:
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-10, 10])
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 플롯을 이미지로 저장
    save_path = 'vis_result_gt/coordinate_systems_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved at {save_path}")
    
    # 저장된 이미지 표시 (선택사항)
    img = mmcv.imread(save_path)
    mmcv.imshow(img)

else:
    print("Sample data token not found for the given image path.")