from collections import defaultdict
from typing import Sequence, Tuple

from torch import int64, zeros, Tensor
from torch.utils.data import Dataset

from .susl_dataset import DatasetFacade, LabeledDatasetFacade


def create_susl_dataset(
    dataset: Dataset, num_labels: float = 0.2, classes_to_hide: Sequence[int] = None
) -> Tuple[Dataset, Dataset, Tensor]:
    # Find ids for each class
    ids = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        ids[label].append(i)
    ids_labeled, ids_unlabeled = [], []
    class_mapper = zeros(len(ids), dtype=int64)
    # Hide classes
    if classes_to_hide is not None:
        for class_to_hide in classes_to_hide:
            ids_unlabeled.extend(ids.pop(class_to_hide))
    # Create ssl
    for v in ids.values():
        size = max(1, int(len(v) * num_labels))
        ids_labeled.extend(v[:size])
        ids_unlabeled.extend(v[size:])
    # Update class mappings
    for i, k in enumerate(sorted(ids.keys())):
        class_mapper[k] = i
    for i, k in enumerate(sorted(classes_to_hide), start=len(ids)):
        class_mapper[k] = i
    # Return facades
    return (
        LabeledDatasetFacade(dataset, ids_labeled, class_mapper=class_mapper),
        DatasetFacade(dataset, ids_unlabeled),
        class_mapper,
    )
