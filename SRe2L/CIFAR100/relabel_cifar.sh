python3 -W ignore relabel_cifar.py \
    --epochs 400 \
    --device $1 \
    --method $4 \
    --output-dir ./save_post_cifar100/ipc50 \
    --syn-data-path $3 \
    --teacher-path 'save/cifar100/resnet18_E200/ckpt.pth' \
    --ipc $2 --batch-size 128
