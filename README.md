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
- for 09/03
    - fix functions
        - (DONE) view mask (visualize/plot_image)
        - (DONE) view image (visualize/plot_mask)
        - (DONE) get predictions (inference/get_prediction_tensor)
        - (DONE) get mask from prediction and overlay (inference/convert_all_predictions_to_mask_and_overlay)
        - (DONE) overlay any mask to image
        - plot manual
        - (DONE) combine foes/fof2 in manual (eval_with_manual/prepare_manual_and_numeric_for_evaluation)
        - (DONE) automate getting metrics of model (eval_with_manual/get_metrics_all_numeric_with_manual)
    - functions to create
        - (DONE) plot/save numeric
    - (DONE) train all models with 2020/2019 data
    - (DONE) get numeric of all models
    - (DONE) compare numeric with manual
        - (DONE) compare numeric of different models against manual
        - (DONE) use rse
    - (DONE) overlay numeric plot with original image
    - bikin surat permohonan magang dan di tambah di awal proposal
    presentation:
        - (DONE but decided not to put) accuracy vs. loss of new models (20, 21, 22)
        - (DONE) overlay of new models
        - (DONE) example of numeric plot over original
        - (DONE) rse, rmse of numeric of models with manual
        - (DONE) rse, rmse of quickscale
- for 09/14
    - (no point) plot manual
    - (DONE) get dice Coefficient of fpn
    - (DONE) get metrics similar to quickscale paper (nighttime/daytime/seasonal)
        - per month, time (use single_eval)
        - per season, day/night (use modified all_eval)
        - all (use all_eval)
    - (DONE) make app to quickly get numeric plot/csv
    - questions for pak varul
        - RMSE, MSE
        - graphs missing 1 month
        - graphs are per month/time?
- for 9/24
    - retrain on more parameters, see how tight segments are
    - provide sample image for website
    - only get fof2, all data (only count when both have values)
    - get metric on all models
        - total
        - per season
        - per time
    because plots and manual are derived differently,
    part of blame can be on creation of plots,
    which we have no control over. however, other times
    there may be a clear indication of a frequency on the
    plot that the model just didn't capture. due to the nature
    of segmentation, model only captures what is on plot already.
    so model can not capture something not there (false positives
    are entirely plots fault).
    model +, manual + = plot is good, model is good (but can be plot bad, model bad by halucinating and accidentally getting true positive)
    model +, manual - = plot is good/bad, model is good/bad (one has to be bad, can't both be bad - both bad means model didn't capture but would register as true negative)
    model -, manual + = plot is good/bad, model is good/bad (one has to be bad, can not be both bad - both bad means model halucinated but would register as true positive)
    model -, manual - = plot is good, model is good (but can be plot bad, model bad by not capturing and accidentally getting true negative)
    ultimately, we don't count false positive/negative as
    could be fault of plot, complicated to tell.
    we also don't count true negatives as the absence of frequencies
    dont necessarily mean frequency is 0 at that time. so we don't
    have a value to compare at those times.
    - presentation
        - introduction
            - goal/purpose
            - alternative methods
        - methods
            - dataset and preparation
            - models, strengths
            - process of getting numeric
        - results
            - training results
            - metrics against manual
        - discussion
            - models and encoders that worked best
            - future work
- for 1/19
    - train on all 2019-some 2020, test on remaining 2020
    - use manuals as annotations
    - train with 3 encoders (for each encoder, keep size same across decoders)


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
    - v17: fpn with 2019/2020 data, 15 epochs
    - v18: deeplabv3, 2019/2020, 15 epochs
    - v19: unet, 2019/2020, 15 epochs
    - v20: fpn, resized images, 2019/2020, 15 Es
    - v21: deeplabv3, resized images, 2019/2020, 15 Es
    - v22: unet, resized images, 2019/2020, 15 Es
    - v23: fpn, resized images, 2019/2020, 10 Es, f1
    - v24: deeplab, resized images, 2019/2020, 10 Es, f1
    - v25: unet, resized images, 2019/2020, 10 Es, f1
    - v26: unet, resized images, 2019 only, 10 Es, f1
    - v27: deeplab, resized images, 2019 only, 10 Es, f1
    - v28: fpn, resized images, 2019 only, 10 Es, f1
    - v29: fpn, resized images, 2019 only, 10 Es, f1, efficientnet-b6
    - v30: deeplab, resized images, 2019 only, 10 Es, f1, efficientnet-b2
    - v31: unet, resized images, 2019 only, 10 Es, f1, efficientnet-b6
    - v32: unet, 2019 only, manual masks
- post pipeline:
    - predict (inference.py)
    - get mask and overlay (inference.py)
    - get numeric from mask (get_numeric.py)
    - prepare manual/numeric (need both because dependent on empty cells from both) (eval_with_manual.py)
    - get rse, rmse (eval_with_manual.py)
    - get specific case metrics (eval_for_paper.py)
- numeric from predictions: the predicted mask has no concept of time. so what is the best way to get data "every 15 minutes"? we could get a linspace of 0-24 (total 96) and then scale it up to the dims of the mask, but what we chose to do was simply get a linspace of 0-mask_dim (total 96).
- quickscale: since time for each fmin/foF2 is different, we will merge_asof fmin/foF2 separately
    - comparing with manual will be on different times because numeric has gaps but qs doesn't, so only filtering based on manual. this shouldn't be a problem because we only want overall accuracy, not accuracy on specific time

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
