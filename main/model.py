import torch
import lightning as L
import segmentation_models_pytorch as smp

class FO2Model(L.LightningModule):
    def __init__(self, architecture, encoder_name, encoder_weights, in_channels, classes, device):
        super().__init__()
        self.save_hyperparameters() # Automatically save hyperparameters so we don't have to pass all arguments at inference
        self.architecture_name = architecture
        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,    # pre-trained weights
            in_channels=in_channels,            # 1 for gray-scale images, 3 for RGB
            classes=classes,                    # number of classes in dataset
        )

        # Getting weights from pre-trained model and setting them as non-trainable
        # Store running statistics to implement custom normalization layer in forward
        # Updated in training but fixed in evaluation
        preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("running_std", torch.tensor(preprocessing_params["std"]).view(1, 3, 1, 1)) # reshapes the tensor so it can be broadcasted across a 3-channel (RGB) image 
        self.register_buffer("running_mean", torch.tensor(preprocessing_params["mean"]).view(1, 3, 1, 1))

        self.loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.threshold = 0.5    # Threshold to calculate IoU score
        
        # For storing loss, IoU info
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.dev = device
        self.model.float()

    def forward(self, image):
        image = (image - self.running_mean) / self.running_std  # Custom normalization
        output = self.model(image)                # Passing image to model to train
        return output 

    def handle_batch(self, batch):
        # Incoming image must have shape (batch, channels, height, width)
        # Incoming mask must have values between 0 and 1
        image, mask, _ = batch

        # Must be true to correctly calculate IoU score
        assert torch.all((mask == 0) | (mask == 1)) # False

        logits_mask = self.forward(image)
        loss = self.loss(logits_mask, mask)
        
        # Calculate IoU score
        outputs = logits_mask.sigmoid()
        outputs = (outputs > self.threshold).float()
        true_positive, false_positive, false_negative, true_negative = smp.metrics.get_stats(outputs.long(), mask.long(), mode="binary")

        return {
            "loss": loss, 
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }

    def handle_epoch_end(self, outputs, stage):
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        true_positive = torch.cat([x["true_positive"] for x in outputs])
        false_positive = torch.cat([x["false_positive"] for x in outputs])
        false_negative = torch.cat([x["false_negative"] for x in outputs])
        true_negative = torch.cat([x["true_negative"] for x in outputs])

        dataset_iou = smp.metrics.iou_score(true_positive, false_positive, false_negative, true_negative, reduction="micro")
        metrics = {
            f"{stage}_loss": mean_loss,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_metrics = self.handle_batch(batch)
        self.training_step_outputs.append(train_metrics)
        return train_metrics

    def on_train_epoch_end(self):
        self.handle_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        return None

    def validation_step(self, batch, batch_idx):
        val_metrics = self.handle_batch(batch)
        self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def on_validation_epoch_end(self):
        self.handle_epoch_end(self.validation_step_outputs, "validation")
        self.validation_step_outputs.clear()
        return None

    def test_step(self, batch, batch_idx):
        test_metrics = self.handle_batch(batch)
        self.test_step_outputs.append(test_metrics)
        return test_metrics

    def on_test_epoch_end(self):
        self.handle_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
        return None

    def configure_optimizers(self):
        # TODO: Unsure of what to use here
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return ([optimizer], [learning_rate_scheduler])

    def predict_step(self, batch, batch_idx):
        inputs, target, filename = batch
        logits_mask = self.forward(inputs)

        probability_mask = logits_mask.sigmoid()
        final_binary_mask = (probability_mask > 0.5).float()
        mask_2d = final_binary_mask.squeeze().cpu()
        mask_array = mask_2d.numpy()
        #print(mask_array[0][0])
        return mask_array

