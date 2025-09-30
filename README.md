foF2 segmentation project for BRIN

files -> dataloader (-> dataset (-> transformations)) -> trainer

Todo:
- research what optimizer/learning rate/scheduler/step size is best
- implement setting in visualize fn to see image, true mask, and prediction mask
- get data to put into dataloader

References:
- Models loaded from SMP
    - https://smp.readthedocs.io/en/latest/quickstart.html
    - https://smp.readthedocs.io/en/latest/models.html
- Encoders in smp
    - https://smp.readthedocs.io/en/latest/encoders.html#choosing-the-right-encoder
- Models used
    - UNET
        - https://arxiv.org/pdf/1505.04597
    - FPN / Feature Pyramid Network
        - https://arxiv.org/pdf/1612.03144
    - DeepLabV3
        - https://arxiv.org/pdf/1706.05587
- Lightning module for more efficient itteration using different architectures
    - https://github.com/Lightning-AI/pytorch-lightning
    - https://medium.com/innovation-res/simplify-your-pytorch-code-with-pytorch-lightning-5d9e4ebd3cfd
    - https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html#training
- Transformations using albumentations library; implemented were inspired by example from Albumentations creators 
    - https://albumentations.ai/docs/
    - https://albumentations.ai/docs/examples/example-kaggle-salt/
    - https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
- Custom normalization with running buffer
    - https://medium.com/data-scientists-diary/understanding-and-effectively-using-register-buffer-in-pytorch-72e6d1c94a95
- Dice Loss as loss function
    - https://arxiv.org/pdf/2312.05391v1
