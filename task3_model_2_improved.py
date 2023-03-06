import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders3 import load_cifar10_3
from trainer3e import Trainer3, compute_loss_and_accuracy


class Model2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # Task 2a - Initialize the neural network
        self.num_classes = num_classes

        # Defining the neural network
        self.feature_extractor = nn.Sequential(
            # First layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True), 
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Second layer no MaxPooling
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True), 
            
            # Third layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True), 
            nn.MaxPool2d(stride=2, kernel_size=2),
           
            # Fourth layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True), 
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Fifth layer
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True), 
            nn.MaxPool2d(stride=2, kernel_size=2),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            # First FC layer
            nn.Linear(2*2*256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            
            # Second FC layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),

            # Third FC layer
            nn.Linear(32, num_classes),
            
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]

        x = self.feature_extractor(x)
        # Flatten
        x = x.view(batch_size, -1)
        x = self.classifier(x)

        out = x

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer3, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_model2.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 3e-2
    early_stop_count = 4
    dataloaders = load_cifar10_3(batch_size)
    model = Model2(image_channels=3, num_classes=10)
    trainer = Trainer3(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    trainer.load_best_model()
    
    create_plots(trainer, "task3_2")

    train, validation, test = dataloaders

    print("---- TRAINING ----")
    train_loss, train_acc = compute_loss_and_accuracy(train, model, nn.CrossEntropyLoss())
    print("---- VALIDATION ----")
    val_loss, val_acc = compute_loss_and_accuracy(validation, model, nn.CrossEntropyLoss())
    print("---- TEST ----")
    test_loss, test_acc = compute_loss_and_accuracy(test, model, nn.CrossEntropyLoss())


if __name__ == "__main__":
    main()