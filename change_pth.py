import torch

def rename_layers(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('encoders.camera.backbone'):
            new_key = key.replace('encoders.camera.backbone', 'img_backbone')
        elif key.startswith('encoders.camera.neck'):
            new_key = key.replace('encoders.camera.neck', 'img_neck')
        elif key.startswith('encoders.camera.vtransform'):
            new_key = key.replace('encoders.camera.vtransform', 'view_transform')
        elif key.startswith('decoder.backbone'):
            new_key = key.replace('decoder.backbone', 'img_backbone_decoder')
        elif key.startswith('decoder.neck'):
            new_key = key.replace('decoder.neck', 'img_neck_decoder')
        elif key.startswith('heads.object'):
            new_key = key.replace('heads.object', 'bbox_head')
        new_state_dict[new_key] = value
    return new_state_dict

# pth 파일 로드
checkpoint = torch.load('pretrained/camera-only-det.pth')

# state_dict 추출
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# 레이어 이름 변경
new_state_dict = rename_layers(state_dict)

# 변경된 state_dict를 checkpoint에 다시 저장
if 'state_dict' in checkpoint:
    checkpoint['state_dict'] = new_state_dict
else:
    checkpoint = new_state_dict

# 변경된 checkpoint를 새 파일로 저장
torch.save(checkpoint, 'pretrained/change_mmdet_final.pth')