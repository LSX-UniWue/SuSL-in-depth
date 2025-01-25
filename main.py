from lightning import Trainer
from torch import Generator, float32, flatten
from torch.nn import Sequential, ReLU, Identity, Flatten, Upsample, Linear, Conv2d
from torch.utils.data import random_split
from torchmetrics import MetricCollection
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor, Compose, ToDtype, Lambda

from data.data_module import SemiUnsupervisedDataModule
from data.utils import create_susl_dataset
from metrics.cluster_and_label import ClusterAccuracy
from networks.gmm_dgm import (
    L2RegularizedGaussianMixtureDeepGenerativeModel,
    EntropyRegularizedGaussianMixtureDeepGenerativeModel,
)
from networks.latent_layer import LatentLayer
from networks.lightning import LightningGMMModel
from networks.losses import GaussianMixtureDeepGenerativeLoss, EntropyGaussianMixtureDeepGenerativeLoss
from networks.misc import Reshape
from networks.variational_layer import GaussianVariationalLayer, BernoulliVariationalLayer


def run_cnn() -> None:
    # Create datasets
    transforms = Compose(
        [
            PILToTensor(),
            ToDtype(float32, scale=True),
            Lambda(lambda x: (x >= 0.5).float()),
        ]
    )

    train_dataset, validation_dataset = random_split(
        MNIST(root="/tmp", train=True, download=True, transform=transforms),
        lengths=[0.8, 0.2],
        generator=Generator().manual_seed(42),
    )
    train_dataset_labeled, train_dataset_unlabeled = create_susl_dataset(
        dataset=train_dataset, num_labels=0.2, classes_to_hide=[5, 6, 7, 8, 9]
    )

    test_dataset = MNIST(root="/tmp", train=False, download=True, transform=transforms)

    datamodule = SemiUnsupervisedDataModule(
        train_dataset_labeled=train_dataset_labeled,
        train_dataset_unlabeled=train_dataset_unlabeled,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=128,
    )
    # Create model
    n_x, n_y, n_z = 28 * 28, 10, 50
    q_y_x_module = Sequential(
        Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Flatten(),
        Linear(in_features=64 * 7 * 7, out_features=n_y),
    )
    p_x_z_module = BernoulliVariationalLayer(
        feature_extractor=Sequential(
            Linear(in_features=n_z, out_features=64 * 7 * 7),
            Reshape((-1, 64, 7, 7)),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
            Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
        ),
        module_init=Conv2d,
        out_channels=1,
        in_channels=1,
        kernel_size=1,
    )
    p_z_y_module = GaussianVariationalLayer(feature_extractor=Identity(), in_features=n_y, out_features=n_z)
    q_z_xy_module = GaussianVariationalLayer(
        feature_extractor=LatentLayer(
            pre_module=Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Flatten(),
            ),
            post_module=Sequential(Linear(in_features=64 * 7 * 7 + n_y, out_features=128), ReLU()),
        ),
        out_features=n_z,
        in_features=128,
    )
    model = EntropyRegularizedGaussianMixtureDeepGenerativeModel(
        n_y=n_y,
        n_z=n_z,
        n_x=n_x,
        q_y_x_module=q_y_x_module,
        p_x_z_module=p_x_z_module,
        p_z_y_module=p_z_y_module,
        q_z_xy_module=q_z_xy_module,
    )
    print(model)
    # Create trainer and run
    lt_model = LightningGMMModel(
        model=model,
        loss_fn=EntropyGaussianMixtureDeepGenerativeLoss(),
        val_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_y, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_y, average="macro"),
            },
            prefix="val_",
        ),
        test_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_y, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_y, average="macro"),
            },
            prefix="test_",
        ),
    )
    trainer = Trainer(max_epochs=10, check_val_every_n_epoch=2)
    trainer.fit(model=lt_model, datamodule=datamodule)
    trainer.test(model=lt_model, datamodule=datamodule)


def run_linear() -> None:
    # Create datasets
    transforms = Compose(
        [
            PILToTensor(),
            ToDtype(float32, scale=True),
            Lambda(lambda x: flatten(x)),
            Lambda(lambda x: (x >= 0.5).float()),
        ]
    )

    train_dataset, validation_dataset = random_split(
        MNIST(root="/tmp", train=True, download=True, transform=transforms),
        lengths=[0.8, 0.2],
        generator=Generator().manual_seed(42),
    )
    train_dataset_labeled, train_dataset_unlabeled = create_susl_dataset(
        dataset=train_dataset, num_labels=0.2, classes_to_hide=[5, 6, 7, 8, 9]
    )

    test_dataset = MNIST(root="/tmp", train=False, download=True, transform=transforms)

    datamodule = SemiUnsupervisedDataModule(
        train_dataset_labeled=train_dataset_labeled,
        train_dataset_unlabeled=train_dataset_unlabeled,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=128,
    )
    # Create model
    n_x, n_y, n_z = 28 * 28, 10, 50
    hidden_dim = 500
    q_y_x_module = Sequential(
        Linear(in_features=n_x, out_features=hidden_dim),
        ReLU(),
        Linear(in_features=hidden_dim, out_features=hidden_dim),
        ReLU(),
        Linear(in_features=hidden_dim, out_features=n_y),
    )
    p_x_z_module = BernoulliVariationalLayer(
        feature_extractor=Sequential(
            Linear(in_features=n_z, out_features=hidden_dim),
            ReLU(),
            Linear(in_features=hidden_dim, out_features=hidden_dim),
            ReLU(),
        ),
        out_features=n_x,
        in_features=hidden_dim,
    )
    p_z_y_module = GaussianVariationalLayer(feature_extractor=Identity(), out_features=n_z, in_features=n_y)
    q_z_xy_module = GaussianVariationalLayer(
        feature_extractor=LatentLayer(
            pre_module=Sequential(
                Linear(in_features=n_x, out_features=hidden_dim),
                ReLU(),
                Linear(in_features=hidden_dim, out_features=hidden_dim),
                ReLU(),
            )
        ),
        out_features=n_z,
        in_features=hidden_dim + n_y,
    )
    model = L2RegularizedGaussianMixtureDeepGenerativeModel(
        n_y=n_y,
        n_z=n_z,
        n_x=n_x,
        q_y_x_module=q_y_x_module,
        p_x_z_module=p_x_z_module,
        p_z_y_module=p_z_y_module,
        q_z_xy_module=q_z_xy_module,
    )
    print(model)
    # Create trainer and run
    lt_model = LightningGMMModel(
        model=model,
        loss_fn=GaussianMixtureDeepGenerativeLoss(gamma=5e-5),
        val_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_y, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_y, average="macro"),
            },
            prefix="val_",
        ),
        test_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_y, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_y, average="macro"),
            },
            prefix="test_",
        ),
    )
    trainer = Trainer(max_epochs=20, check_val_every_n_epoch=1)
    trainer.fit(model=lt_model, datamodule=datamodule)
    trainer.test(model=lt_model, datamodule=datamodule)


if __name__ == "__main__":
    run_cnn()
