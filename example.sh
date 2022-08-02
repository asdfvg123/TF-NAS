# For searching, taking the following script as an example
CUDA_VISIBLE_DEVICES=0 python -W ignore -u train_search.py \
	--img_root "/work/08467/yjeon/longhorn/imagenet" \
	--train_list "./dataset/ImageNet-100-effb0_train_cls_ratio0.8.txt" \
	--val_list "./dataset/ImageNet-100-effb0_val_cls_ratio0.8.txt" \
	--lookup_path "./latency_pkl/latency_gpu.pkl" \
	--save "./checkpoints" \
	--print_freq 100 \
	--workers 4 \
	--epochs 90 \
	--batch_size 16\
	--w_lr 0.025 \
	--w_mom 0.9 \
	--w_wd 1e-5 \
	--a_lr 0.01 \
	--a_wd 5e-4 \
	--grad_clip 5.0 \
	--T 5.0 \
	--T_decay 0.96 \
	--num_classes 100 \
	--lambda_lat 0.1 \
	--target_lat 15.0 \
	--cost_type "add" \
	--note "TF-NAS-lam0.1-lat15.0-gpu, fresh cost" \
	--checkpoint_path "/work/08467/yjeon/longhorn/myproject/TF-NAS/checkpoints/search-20220309-205710-TF-NAS-lam0.1-lat15.0-gpu-fresh-cost/" \
	--checkpoint_epoch 36

# # After searching, you can parse the searched architecture by
# CUDA_VISIBLE_DEVICES=0 python -u parsing_model.py \
# 	--model_path "Searched Model Path" \
# 	--save_path "./model.config" \
# 	--lookup_path "./latency_pkl/latency_gpu.pkl" \
# 	# --print_lat


# # Use --model_path to parse the searched architecture
# CUDA_VISIBLE_DEVICES=0,1 python -u train_eval.py \
# 	--train_root "Your ImageNet Train Set Path" \
# 	--val_root "Your ImageNet Val Set Path" \
# 	--train_list "ImageNet Train List" \
# 	--val_list "ImageNet Val List" \
# 	--model_path "Searched Model Path" \
# 	--save "./checkpoints" \
# 	--print_freq 100 \
# 	--workers 16 \
# 	--epochs 250 \
# 	--batch_size 512 \
# 	--lr 0.2 \
# 	--momentum 0.9 \
# 	--weight_decay 1e-5 \
# 	--grad_clip 5.0 \
# 	--label_smooth 0.1 \
# 	--num_classes 1000 \
# 	--dropout_rate 0.2 \
# 	--drop_connect_rate 0.2 \
# 	--note "TF-NAS-lat15.0-gpu"


# # Or use --config_path to parse the searched architecture
# CUDA_VISIBLE_DEVICES=0,1 python -u train_eval.py \
# 	--train_root "Your ImageNet Train Set Path" \
# 	--val_root "Your ImageNet Val Set Path" \
# 	--train_list "ImageNet Train List" \
# 	--val_list "ImageNet Val List" \
# 	--config_path "./model.config" \
# 	--save "./checkpoints" \
# 	--print_freq 100 \
# 	--workers 16 \
# 	--epochs 250 \
# 	--batch_size 512 \
# 	--lr 0.2 \
# 	--momentum 0.9 \
# 	--weight_decay 1e-5 \
# 	--grad_clip 5.0 \
# 	--label_smooth 0.1 \
# 	--num_classes 1000 \
# 	--dropout_rate 0.2 \
# 	--drop_connect_rate 0.2 \
# 	--note "TF-NAS-lat15.0-gpu"


# # Or employ Automatic Mixed Precision
# CUDA_VISIBLE_DEVICES=0,1 python -u train_eval_amp.py \
# 	--train_root "Your ImageNet Train Set Path" \
# 	--val_root "Your ImageNet Val Set Path" \
# 	--train_list "ImageNet Train List" \
# 	--val_list "ImageNet Val List" \
# 	--config_path "./model.config" \
# 	--save "./checkpoints" \
# 	--print_freq 100 \
# 	--workers 16 \
# 	--epochs 250 \
# 	--batch_size 512 \
# 	--lr 0.2 \
# 	--momentum 0.9 \
# 	--weight_decay 1e-5 \
# 	--grad_clip 5.0 \
# 	--label_smooth 0.1 \
# 	--num_classes 1000 \
# 	--dropout_rate 0.2 \
# 	--drop_connect_rate 0.2 \
# 	--opt_level "O1" \
# 	--note "TF-NAS-lat15.0-gpu"


# # Or employ Automatic Mixed Precision + DistributedDataParallel
# CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 train_eval_amp.py \
# 	--train_root "Your ImageNet Train Set Path" \
# 	--val_root "Your ImageNet Val Set Path" \
# 	--train_list "ImageNet Train List" \
# 	--val_list "ImageNet Val List" \
# 	--config_path "./model.config" \
# 	--save "./checkpoints" \
# 	--print_freq 100 \
# 	--workers 16 \
# 	--epochs 250 \
# 	--batch_size 512 \
# 	--lr 0.2 \
# 	--momentum 0.9 \
# 	--weight_decay 1e-5 \
# 	--grad_clip 5.0 \
# 	--label_smooth 0.1 \
# 	--num_classes 1000 \
# 	--dropout_rate 0.2 \
# 	--drop_connect_rate 0.2 \
# 	--opt_level "O1" \
# 	--note "TF-NAS-lat15.0-gpu"


# # After training, you can test the trained model by
# CUDA_VISIBLE_DEVICES=0 python -u test.py \
# 	--val_root "Your ImageNet Val Set Path" \
# 	--val_list "ImageNet Val List" \
# 	--config_path "./model.config" \
# 	--weights "Pretrained Weights"
