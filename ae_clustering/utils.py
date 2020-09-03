from typing import Dict


def read_data(corpus_file: str, label_column: int, document_start: int) -> Dict:
    """
    Returns a <key, value> pair of the loaded dataset
    where the key is the text data and the value is the data label.

    Parameters
    ----------
    corpus_file: str
        The filename of the dataset to load.
    label_column: int
        The column number of the dataset label (zero-indexed).
    document_start: int
        The number of columns in the dataset.

    Returns
    -------
    dataset: Dict
        The <key, value> pair representing the text data and their labels.
    """
    dataset = {}
    with open(corpus_file, "r", encoding="utf-8") as text_data:
        for line in text_data:
            columns = line.strip().split(maxsplit=document_start)
            text = columns[-1]
            label = int(columns[label_column].strip("__label__"))
            dataset[text] = label
    return dataset
