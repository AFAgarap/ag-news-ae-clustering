import numpy as np
import torch

from ae_clustering.utils import clustering_accuracy, create_dataloader, load_dataset


def test_clustering_accuracy():
    vector_a = np.array([1, 2, 3, 4, 5, 6, 7])
    vector_b = np.array([1, 2, 1, 4, 1, 6, 7])
    expected_output = 0.7142857142857143
    output = clustering_accuracy(vector_a, vector_b)
    assert np.allclose(output, expected_output)


def test_create_dataloader():
    batch_size = 64
    (train_vectors, train_labels, _) = load_dataset("data/ag_news.train")
    (test_vectors, test_labels, _) = load_dataset("data/ag_news.test")
    train_loader = create_dataloader(
        features=train_vectors, labels=train_labels, batch_size=batch_size
    )
    test_loader = create_dataloader(
        features=test_vectors, labels=test_labels, batch_size=batch_size
    )
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    for batch_features, batch_labels in train_loader:
        assert isinstance(batch_features, torch.Tensor)
        assert isinstance(batch_labels, torch.Tensor)
        assert batch_features.size() == (batch_size, 2000)
        break
    for batch_features, batch_labels in test_loader:
        assert isinstance(batch_features, torch.Tensor)
        assert isinstance(batch_labels, torch.Tensor)
        assert batch_features.size() == (batch_size, 2000)
        break


def test_load_dataset():
    (train_vectors, train_labels, _) = load_dataset("data/ag_news.train")
    (test_vectors, test_labels, _) = load_dataset("data/ag_news.test")
    assert len(train_vectors.shape) == 2
    assert train_vectors.shape == (119843, 2000)
    assert len(train_labels.shape) == 1
    assert train_labels.shape == (119843,)
    assert len(test_vectors.shape) == 2
    assert test_vectors.shape == (7600, 2000)
    assert len(test_labels.shape) == 1
    assert test_labels.shape == (7600,)
