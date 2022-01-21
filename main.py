import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import logging
import time

from dataset import CIFar10
from data_loader import load_data, load_data2
from trainer.train import Trainer
from trainer.callbacks import TensorBoardCallback, LoggerCallback
from trainer.eval import Evaluator
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from model.model import *


_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


logging.basicConfig(
    format="%(asctime)s [%(name)s: %(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("cnn.py")

time = time.strftime("%d%H%M", time.localtime())


def main():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info(
            "CUDA version: [ %s ] | CUDA DEVICE: [ %s ]",
            torch.version.cuda,
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
    device = torch.device("cuda" if use_cuda else "cpu")

    train_X, train_y, X_test, y_test, labels = load_data2()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    for batch_size in [32, 64, 128]:
        for lr in [0.1, 0.01, 0.001]:
            train_set = CIFar10(train_set=(train_X, train_y), transform=transform)
            test_set = CIFar10(train_set=(X_test, y_test), transform=transform)

            trainloader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=0
            )

            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=True, num_workers=0
            )

            model = models.resnet50(pretrained=True)

            loss_fn = torch.nn.CrossEntropyLoss()
            logging.info(
                "[ lr: %f ] | [ batch_size: %d ]",
                lr,
                batch_size
            )
            optimizer = optim.SGD(model.parameters(), lr=lr)

            output_dir = f"runs/test{time}/LR {lr} bsize {batch_size}"
            summary_writer = SummaryWriter(output_dir)

            # Initialize Callbacks
            training_callbacks = [
                TensorBoardCallback(
                    log_interval=50,
                    summary_writer=summary_writer,
                    tag_msg="/train"
                ),
                LoggerCallback(
                    log_interval=50,
                )
            ]

            trainer = Trainer(
                data_loader=trainloader,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                callbacks=training_callbacks,
            )

            trainer.fit_model(5)

    eval_callbacks = []
    evaluator = Evaluator(
        data_loader=testloader,
        model=model,
        loss_fn=loss_fn,
        device=device,
        callbacks=eval_callbacks,
    )

    evaluator.eval()


if __name__ == "__main__":
    main()
