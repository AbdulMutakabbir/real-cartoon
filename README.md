# real-cartoon

This repo contains code to build models to predict if an image is of a real person or cartoon.

Is essential to download the dataset and place it in the `dataset` directory befoer strating to run the code.
The dataset can be found at `http://cvit.iiit.ac.in/images/Projects/cartoonFaces/IIIT-CFW1.0.zip`.

## Requirements
- git
- PyTorch
- Jupyter
- GPU (optional)

## Instructions
1. Clone the repo: 
```
git clone https://github.com/AbdulMutakabbir/real-cartoon.git
```
2. Install the requirements
```
pip install -r requirements.txt
```
3. Start Jupyter server

## Model Comparision
Two transfer learning models were chosen. They are:
- ResNet18
- ShuffleNet

These models were chosen for the model size and complexity.

> ResNet performed best with `95% accuracy`

> SuffleNet had `90% accuracy`
