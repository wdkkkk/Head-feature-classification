## Installation

For installation,run 
```
pip install -r requirements.txt
```
## Split the dataset

To split the dataset, you should first copy the dataset into "data_face_imgs" and then ran split_dataset.ipynb in the floder.

## Training
```sh
# Task1
python train.py --config configs/train.json --task smile
python train.py --config configs/train.json --pretrained --task smile
# Task2
python train.py --config configs/train.json --task hair
python train.py --config configs/train.json --pretrained --task hair
# Task3
python train.py --config configs/train.json --task all
python train.py --config configs/train.json --pretrained --task all
```
You could see other configs in the code. 
## Testing
To test your model, you should first find the model's .pth file in output folder and copy it into config file. Then run:
```sh
#Task1
python test.py --config configs/test1.json
#Task2
python test.py --config configs/test1.json
#Task3
python test.py --config configs/test1.json

```

