import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
import torch
from trainer import Trainer, compute_loss_and_accuracy

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    #Added this part to change the data from tensor : gpu to cpu for plotting
    if torch.cuda.is_available():
        d1 = trainer.validation_history
        for key in d1['loss']:
            if isinstance(d1['loss'][key], torch.Tensor):
                d1['loss'][key] = d1['loss'][key].item()
        for key in d1['accuracy']:
            d1['accuracy'][key] = float(d1['accuracy'][key])
        
        utils.plot_loss(trainer.validation_history["loss"], label="Validation loss", npoints_to_average=1) #prøve npoints_to_average=10
    else:
        utils.plot_loss(trainer.validation_history["loss"], label="Validation loss", npoints_to_average=1) #prøve npoints_to_average=10
    #the function above does not generate an error message, but the plot for validation loss is not shown in the plot, why is that?
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    #her er problemet som rammes av cpu/gpu feil
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}.png"))
    plt.show()
