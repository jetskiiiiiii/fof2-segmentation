foF2 segmentation project for BRIN

files -> dataloader (-> dataset (-> transformations)) -> trainer

Todo:
- research what optimizer/learning rate/scheduler/step size is best
- use learning rate scheduler
- implement setting in visualize fn to see image, true mask, and prediction mask
- (DONE) get data to put into dataloader
- implement metrics (IoU, Dice Coefficient, Pixel Accuracy)
    - val should calculate metrics at each epoch
- for 08/17
    - experiment with batch size
    - compare accuracy between augmented vs. non-augmented
    - see model architecture 
    - try different architecture
    - plot with stacked bar chart
    - when plotting, be mindful of empty graphs
- for 08/21
    - finish roboflow annotations
    - train on bigger dataset
    - turn mask into numeric
    - draw masks onto graph for models
- for 08/28
    - train with plotting masks
    - annotate roboflow
    - get numeric of predictions
- for 09/
    - compare numeric with manual
        - compare numeric of different models against manual
        - use rse
    - overlay numeric plot with original image
    - fix functions
        - move and rename image/masks
        - view mask
        - view image
        - overlay mask to image
    - bikin surat permohonan magang dan di tambah di awal proposal

Notes:
- losses
    - Dice Loss: Ideal for handling imbalanced classes by focusing on overlap.
    - Tversky Loss: Adds flexibility by adjusting the penalty on false positives vs. false negatives.
    - Focal Loss: Useful for segmenting small or rare objects by down-weighting easy examples.
- unet_models
    - v5 - change LR to 1e-4, add Cosine Annealing scheduler
    - v9 - change mask extraction method
    - v10 - more changes to mask extraction to account for values other than 0 and 1
- why ground truth produces a weird mask:
    - differences in scatter plot scale and fti image scale
    - process of making fti and making numeric values differ
- model versions
    - v11: 32 batch
    - v12: 8 batch
    - v13: 64 batch
    - v14: no transform
    - v15: deeplab
    - v16: fpn

Concerns:
- what to do with fmin without fof2?

References:
- Image Segmentation Practices
    - https://medium.com/@heyamit10/pytorch-segmentation-models-a-practical-guide-5bf973a32e30
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
- Accuracy metrics
    - Binary accuracy
        - https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#binaryaccuracy
    - IoU Metrics
        - https://smp.readthedocs.io/en/latest/metrics.html#segmentation_models_pytorch.metrics.functional.get_stats
- Logging metrics
    - https://lightning.ai/docs/torchmetrics/v1.8.2/pages/lightning.html
    - https://lightning.ai/docs/pytorch/stable/extensions/logging.html
