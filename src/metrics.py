from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs) -> MetricCollection:
    """
    Retrieves and initializes a collection of metrics for evaluation.

    Args:
        ``**kwargs``: Keyword arguments to configure the metrics.

    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
        },
    )
