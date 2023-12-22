cd ..
CUDA_VISIBLE_DEVICES=7 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/train.json --task all
CUDA_VISIBLE_DEVICES=7 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/train.json --pretrained --task all
CUDA_VISIBLE_DEVICES=7 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/train.json --task hair
CUDA_VISIBLE_DEVICES=7 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/train.json --pretrained --task hair
CUDA_VISIBLE_DEVICES=5 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/train.json --task smile
CUDA_VISIBLE_DEVICES=7 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/train.json --pretrained --task smile