import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
import torch
#from dataloaders import load_cifar10
from trainer3a import Trainer3, compute_loss_and_accuracy


class Model1(nn.Module):

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
        num_filters = [32, 64, 128]  # Set number of filters in first conv layer, 32 is the number of filters in the first convolutional layer, 
        #64 is for the second, 128 or the third
        self.num_classes = num_classes

        # Defining the neural network
        self.feature_extractor = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[0]),
            nn.Hardswish(inplace=True), #TODO inplace=true
            # First max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Second convolutional layer
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[0],#har regnet at dette er 32 men de setter den til 64
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[0]),
            nn.Hardswish(inplace=True), #TODO inplace=true

            # Second max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Third convolutional layer
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[0]),
            nn.Hardswish(inplace=True), #TODO inplace=true

            # Third max pool
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
            nn.Linear(32*32*32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
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
    
    def save_model(self, PATH):
        torch.save(self.state_dict(), PATH)


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
    plt.savefig(plot_path.joinpath(f"{name}_model1.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 7e-2
    early_stop_count = 5
    dataloaders = load_cifar10(batch_size)
    model = Model1(image_channels=3, num_classes=10)
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
    create_plots(trainer, "3a")

    train, validation, test = dataloaders

    print("---- TRAINING ----")
    train_loss, train_acc = compute_loss_and_accuracy(train, model, nn.CrossEntropyLoss())
    print("---- VALIDATION ----")
    val_loss, val_acc = compute_loss_and_accuracy(validation, model, nn.CrossEntropyLoss())
    print("---- TEST ----")
    test_loss, test_acc = compute_loss_and_accuracy(test, model, nn.CrossEntropyLoss())


if __name__ == "__main__":
    main()