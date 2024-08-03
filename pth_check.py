import torch
import sys
import io

# 표준 출력을 파일로 리다이렉트하기 위한 클래스
class OutputRedirector(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 출력을 파일로 리다이렉트
sys.stdout = OutputRedirector("checkpoint_comparison.txt")

# BEVFusion 체크포인트 로드
bevfusion_ckpt = torch.load('/media/spalab/sdb/kypark/SeparateModalityFusion/pretrained/camera-only-det.pth', map_location='cpu')

# MMDetection3D 체크포인트 로드
mmdet3d_ckpt = torch.load('/media/spalab/sdb/kypark/SeparateModalityFusion/work_dirs/dh/bevfusion_camera_only_centerhead_0724/epoch_5.pth', map_location='cpu')
print("BEVFusion checkpoint keys:")
for k in bevfusion_ckpt['state_dict'].keys():
    print(k)

print("\nMMDetection3D checkpoint keys:")
for k in mmdet3d_ckpt['state_dict'].keys():
    print(k)

# 키 구조 비교
bevfusion_keys = set(bevfusion_ckpt['state_dict'].keys())
mmdet3d_keys = set(mmdet3d_ckpt['state_dict'].keys())

print("\nKeys in BEVFusion but not in MMDetection3D:")
print(bevfusion_keys - mmdet3d_keys)

print("\nKeys in MMDetection3D but not in BEVFusion:")
print(mmdet3d_keys - bevfusion_keys)

# 공통 키의 shape 비교
print("\nComparing shapes of common keys:")
common_keys = bevfusion_keys.intersection(mmdet3d_keys)
for key in common_keys:
    bev_shape = bevfusion_ckpt['state_dict'][key].shape
    mm_shape = mmdet3d_ckpt['state_dict'][key].shape
    if bev_shape != mm_shape:
        print(f"{key}: BEVFusion shape {bev_shape}, MMDetection3D shape {mm_shape}")

# 표준 출력을 원래대로 복구
sys.stdout = sys.stdout.terminal

print("Comparison completed. Results saved in 'checkpoint_comparison.txt'")