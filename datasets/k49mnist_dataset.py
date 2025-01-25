import os

from numpy import load
from torch import from_numpy
from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_url, check_integrity


class K49MNIST(MNIST):
    """`Kuzushiji49-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``K49MNIST/raw/k49-train-imgs.npz``
            and  ``K49MNIST/raw/k49-test-imgs.npz`` exist.
        train (bool, optional): If True, creates dataset from ``k49-train-imgs.npz``,
            otherwise from ``k49-test-imgs.npz``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/k49/"]

    resources = [
        ("k49-train-imgs.npz", "7ac088b20481cf51dcd01ceaab89d821"),
        ("k49-train-labels.npz", "44a8e1b893f81e63ff38d73cad420f7a"),
        ("k49-test-imgs.npz", "d352e201d846ce6b94f42c990966f374"),
        ("k49-test-labels.npz", "4da6f7a62e67a832d5eb1bd85c5ee448"),
    ]

    classes = [
        "\u3042",
        "\u3044",
        "\u3046",
        "\u3048",
        "\u304a",
        "\u304b",
        "\u304d",
        "\u304f",
        "\u3051",
        "\u3053",
        "\u3055",
        "\u3057",
        "\u3059",
        "\u305b",
        "\u305d",
        "\u305f",
        "\u3061",
        "\u3064",
        "\u3066",
        "\u3068",
        "\u306a",
        "\u306b",
        "\u306c",
        "\u306d",
        "\u306e",
        "\u306f",
        "\u3072",
        "\u3075",
        "\u3078",
        "\u307b",
        "\u307e",
        "\u307f",
        "\u3080",
        "\u3081",
        "\u3082",
        "\u3084",
        "\u3086",
        "\u3088",
        "\u3089",
        "\u308a",
        "\u308b",
        "\u308c",
        "\u308d",
        "\u308f",
        "\u3090",
        "\u3091",
        "\u3092",
        "\u3093",
        "\u309d",
    ]

    def download(self) -> None:
        """Download the K49MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                download_url(url=url, root=self.raw_folder, filename=filename, md5=md5)

    def _check_exists(self) -> bool:
        return all(check_integrity(os.path.join(self.raw_folder, url)) for url, _ in self.resources)

    def _load_data(self):
        image_file = f"k49-{'train' if self.train else 'test'}-imgs.npz"
        with load(os.path.join(self.raw_folder, image_file)) as file:
            data = from_numpy(file["arr_0"])

        label_file = f"k49-{'train' if self.train else 'test'}-labels.npz"
        with load(os.path.join(self.raw_folder, label_file)) as file:
            targets = from_numpy(file["arr_0"])

        return data, targets
