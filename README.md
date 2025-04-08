## Install
create a python environment and run the following command to install all requirements
```
pip install torch torchvision opencv-python numpy pytorch-lightning segmentation-models-pytorch scikit-image glob2 scikit-learn
```

## Usage
Download the model from [here](https://drive.google.com/file/d/1_U9YAKFDTgyjLzkLygomZDaGYlzNHIBn/view?usp=sharing) and run the model with
```
python run_model.py --input=./input/002.jpg --output=./output/002.png
```
This code will save the output as .png files for visual inspection and as a .xlsx file for data recording. If the model is not in the TowardsRoxasDL folder specify the path with `--model=path_to_model`

Note: This script will run the model only on the CPU which is significantly slower than on the GPU. This is done to make it as accessible as possible for everyone to try out if the model will work on his data.
```
