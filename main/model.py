import torch
import lightning as L
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryAccuracy

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
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.dev = device
        self.model.float()

    def forward(self, image):
        image = (image - self.running_mean) / self.running_std  # Custom normalization
        mask = self.model(image)                # Passing image to model to train
        return mask

    def handle_batch(self, batch, stage):
        # Incoming image must have shape (batch, channels, height, width)
        # Incoming mask must have values between 0 and 1
        image, mask = batch

        logits_mask = self.forward(image)
        loss = self.loss(logits_mask, mask)

        if stage == "train":
            acc = self.train_acc(logits_mask, mask)
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        if stage == "val": 
            acc = self.val_acc(logits_mask, mask)
            self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss, acc         # During eval stage, also return accuracy

        if stage == "test":
            acc = self.test_acc(logits_mask, mask)
            self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss, acc

    def training_step(self, batch, batch_idx):
        train_loss = self.handle_batch(batch, "train")
        return  train_loss 

    def validation_step(self, batch, batch_idx):
        val_loss, val_acc = self.handle_batch(batch, "val")
        return  val_loss, val_acc 

    def test_step(self, batch, batch_idx):
        test_loss, test_acc = self.handle_batch(batch, "test")
        return test_loss, test_acc

    def configure_optimizers(self):
        # TODO: Unsure of what to use here
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        #learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer

    def predict_step(self, batch):
        inputs, target = batch
        logits_mask = self.forward(inputs)
        #probabilities = torch.sigmoid(logits_mask)
        #binary_mask = (probabilities > 0.5).int()
        return logits_mask
        return binary_mask

