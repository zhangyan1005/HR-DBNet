CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_hrnet_thre.yaml --resume /home/math-tr/jzy/DB/models/model_epoch_865_hrnet18 --box_thresh 0.55
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_hrnet32_thre.yaml --resume /home/math-tr/jzy/DB/models/ic15_HRNet32  --box_thresh 0.55 --speed
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_hrnet32_thre.yaml --resume /home/math-tr/jzy/DB/models/ic15_HRNet32 --box_thresh 0.55

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_hr32_thre.yaml --resume /home/math-tr/jzy/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/hrnet32/L1BalanceCELoss/model/model_epoch_636_minibatch_400000 --polygon --box_thresh 0.55
