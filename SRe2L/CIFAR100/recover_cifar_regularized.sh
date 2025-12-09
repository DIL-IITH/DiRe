python3 -W ignore recover_cifar_regularized.py \
--arch-name "resnet18" \
--arch-path 'save/cifar100/resnet18_E200/ckpt.pth' \
--exp-name "cifar100_regularized" \
--batch-size 50 \
--start_class_id $1 \
--IPC $2 \
--device $3 \
--multiplier $4 \
--ipc-start $5 \
--lr 0.1 \
--iteration 1000 \
--r-bn 1.0 \
--store-best-images 


