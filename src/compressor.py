from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def compressor_performance_calculator(original_dim, compressed_dim, predictions, targets) -> tuple[float, float, float]:
    return (
        compressed_dim / original_dim,
        mean_absolute_error(targets, predictions),
        mean_absolute_percentage_error(targets, predictions),
    )
