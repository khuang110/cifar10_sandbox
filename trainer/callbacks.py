import logging
from abc import ABC, abstractmethod

import torch
import torchvision


logger = logging.getLogger("callbacks.py")

logging.basicConfig(
    format="%(asctime)s [%(name)s: %(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class Callback(ABC):
    def __init__(self):
        """ Constructor. """
        self._model = None
        self._optimizer = None
        self._output_dir = None
        self._lr_scheduler = None
        self._batch_size = None

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    def before_start(self):
        """Called before first epoch."""
        pass

    def after_stop(self):
        """Called after last epoch."""
        pass

    def _check_property(self, prop_name: str):
        """Check that a property has been set and is not None.

        attr_name: Name of attribute to check.
        """
        if not hasattr(self, prop_name) or prop_name is None:
            raise RuntimeError(f"{prop_name} needs to be set in {type(self).__name__}")

    @property
    def model(self) -> torch.nn.Module:
        self._check_property("_model")
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        """Setter for model property.

        model: PyTorch model.
        """
        self._model = model

    @property
    def optimizer(self):
        self._check_property("_optimizer")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer

    @property
    def lr_scheduler(self):
        self._check_property("_lr_scheduler")
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler):
        self._lr_scheduler = lr_scheduler

    @property
    def batch_size(self):
        self._check_property("_batch_size")
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size


class LoggerCallback(Callback):
    def __init__(self, log_interval: int):
        super(LoggerCallback, self).__init__()
        self._log_interval = log_interval

    def __call__(self, global_step: int, total_steps: int, avg_loss: float, **kwargs):
        if global_step % self._log_interval == 0:
            logger.info(
                "Epoch: %d/%d | avg loss: %f",
                global_step + 1,
                total_steps + 1,
                avg_loss,
            )


class TensorBoardCallback(Callback):
    def __init__(self, summary_writer, log_interval: int, tag_msg=None):
        super(TensorBoardCallback, self).__init__()
        self._summary_writer = summary_writer
        self._log_interval = log_interval

        self._tag_msg = tag_msg if tag_msg is not None else ""

    def __call__(
            self,
            global_step: int,
            total_steps: int,
            loss: float,
            X: torch.Tensor,
            y: torch.Tensor,
            predicted: torch.Tensor,
            **kwargs
    ):
        if global_step % self._log_interval == 0:
            # Log the loss in tensorboard
            self._summary_writer.add_scalar(
                tag=f"loss{self._tag_msg}", scalar_value=loss, global_step=global_step
            )

            # Number of correct predicitons
            num_correct = (predicted == y).sum().item()
            # Calculate accuracy and add to tensorboard
            self._summary_writer.add_scalar(
                tag=f"accuracy{self._tag_msg}",
                scalar_value=num_correct / len(predicted),
                global_step=global_step,
            )

        # Log on the last step "Could be replaced with after_stop() method"
        if global_step > 0 and global_step % (total_steps - 1) == 0:
            # Add images that are in the last batch into tensorboard
            img_grid = torchvision.utils.make_grid(X)
            self._summary_writer.add_image('cifar10 images', img_grid / 2 + 0.5)  # Unnormalize images

            # Add Hparams to tensorboard
            num_correct = (predicted == y).sum().item()
            self._summary_writer.add_hparams(
                {'lr': self.optimizer.param_groups[0]['lr'], 'batch_size': self.batch_size},
                {f"accuracy{self._tag_msg}": num_correct / len(predicted),
                 f'loss{self._tag_msg}': loss
                 },
            )

            # Graphical view of model used
            self._summary_writer.add_graph(self.model, X)

            # Add histogram of first fully connected layer
            self._summary_writer.add_histogram(
                'layer1/weight',
                self.model._modules['layer1'][0]._modules['bn1'].weight.data,
                global_step
            )
