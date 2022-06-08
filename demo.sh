# db
CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/ic15_resnet18_thre.yaml --image_path datasets/icdar2015/train_images/img_33.jpg --resume /home/math-tr/jzy/DB/models/ic15_res18_4_1200 --polygon  --box_thresh 0.6 --visualize
CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --visualize
# tram-db
CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/ic15_hrnet18_thre.yaml  --image_path datasets/icdar2015/train_images/img_35.jpg --resume /home/math-tr/jzy/DB/models/ic15pre_hrnet_smu_tmask --polygon --box_thresh 0.6 --visualize
# total-text
CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_hr18_thre.yaml --image_path datasets/total_text/train_images/img11.jpg --resume /home/math-tr/jzy/DB/models/total_hrnet18_smu_mask_1 --polygon --box_thresh 0.7 --visualize