from ae_clustering.utils import clustering_accuracy, create_dataloader, load_dataset


def test_clustering_accuracy():
    pass


def test_create_dataloader():
    pass


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
