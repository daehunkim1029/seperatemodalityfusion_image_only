import torch
import sys
import io

# 출력을 캡처하기 위한 StringIO 객체 생성
output_capture = io.StringIO()
sys.stdout = output_capture

# .pth 파일 경로
path = 'pretrained/change_mmdet_final.pth'
# 파일 로드
checkpoint = torch.load(path)

# state_dict 확인
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("State Dict 내용:")
    for key, value in state_dict.items():
        print(f"Layer: {key}")
        print(f"Type: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"Shape: {value.shape}")
        print("--------------------")
else:
    print("This file does not contain a state_dict.")

# 표준 출력 복원
sys.stdout = sys.__stdout__

# 캡처된 출력을 가져옴
output_content = output_capture.getvalue()

# 출력 내용을 파일로 저장
with open('change_mmdet_structure.txt', 'w') as f:
    f.write(output_content)

print("출력 내용이 'model_structure.txt' 파일로 저장되었습니다.")