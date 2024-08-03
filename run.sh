#work_dir=work_dirs/masking_strategy/version2/image_head_2query_new_small
#bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_small.py 4 --work-dir $work_dir 
# for failures in 'lidar_drop' 'camera_view_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion'
# do
#   bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_with_imghead_small/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
# done

# work_dir=work_dirs/masking_strategy/version2/image_head_2query_new_small_smt
# #bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_small_smt.py 4 --work-dir $work_dir
# for failures in 'lidar_drop'  'camera_view_drop'  'limited_fov' 'object_failure' 'beam_reduction' 'occlusion'
# do
#    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_with_imghead_small/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures/epoch_5
# done
#/media/spalab/sdb/kypark/SeparateModalityFusion/work_dirs/dh/bevfusion_camera_only_centerhead_0724/epoch_5.pth 

#work_dir = work_dirs/dh/bevfusion_camera_only_centerhead_0724
#bash tools/dist_test.sh projects/BEVFusion/configs/bevfusion_camera_only.py  pretrained/change_mmdet_final.pth 4 --work-dir work_dirs/dh/bevfusion_camera_only_centerhead_0729


#bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_camera_only_final.py 1 --work-dir work_dirs/dh/bevfusion_camera_only_final_0730__real_final_tlqkf
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_camera_only_final.py 4 --work-dir work_dirs/dh/bevfusion_camera_only_final_0804_900+1600 

#bash tools/dist_test.sh projects/BEVFusion/configs/bevfusion_camera_only_final.py work_dirs/dh/bevfusion_camera_only_final_0730__real_final_tlqkf/epoch_18.pth  1 --work-dir work_dirs/dh/bevfusion_camera_only_final_0730__real_final_tlqkf