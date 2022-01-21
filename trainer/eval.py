import logging
import torch
from torch.utils.data.dataloader import DataLoader as DataLoader

logger = logging.getLogger("eval.py")

logging.basicConfig(
    format="%(asctime)s [%(name)s: %(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class Evaluator:
    def __init__(
        self,
        data_loader: DataLoader,
        model: torch.nn.Module,
        loss_fn,
        device: torch.device = None,
        output_dir=None,
        callbacks=None,
    ):
        self._data_loader = data_loader
        self._model = model.to(device)
        self._device = device
        self._loss_fn = loss_fn
        self._output_dir = output_dir

        if callbacks is not None:
            self._callbacks = callbacks.copy()

            # Set Callback properties
            for callback in self._callbacks:
                callback.model = self._model
                callback.batch_size = self._data_loader.batch_size
                callback.total_steps = len(self._data_loader)
        else:
            self._callbacks = None

    @torch.no_grad()    # We don't need to calculate gradient
    def eval(self):
        total_samples = 0
        correct = 0
        for global_step, batch in enumerate(self._data_loader):
            X, y = batch[0].to(self._device), batch[1].to(self._device)
            X = torch.as_tensor(X)
            y = torch.as_tensor(y)

            # calculate outputs by running images through the network
            outputs = self._model(X)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total_samples += y.size(0)
            correct += (predicted == y).sum().item()

            if self._callbacks is not None:
                for callback in self._callbacks:
                    callback(
                        global_step=global_step,
                        X=X,
                        y=y,
                    )
        logging.info(
            "Accuracy of the network on the %d test images: %f.2",
            total_samples,
            (100 * correct // total_samples)
        )