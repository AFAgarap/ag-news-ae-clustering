from ae_clustering.utils import clustering_accuracy, create_dataloader, load_dataset


def test_clustering_accuracy():
    pass


def test_create_dataloader():
    pass


def test_load_dataset():
    (train_vectors, train_labels, vectorizer) = load_dataset("data/ag_news.train")
    assert len(train_vectors.shape) == 2
    assert train_vectors.shape == (119843, 2000)
    assert len(train_labels.shape) == 1
    assert train_labels.shape == (119843,)
