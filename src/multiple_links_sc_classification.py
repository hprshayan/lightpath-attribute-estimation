import pathlib

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from src.constants import (
    COMPRESSION_METHOD,
    DISTANCE_FEATURE,
    EXP_DIR,
    LINK_LENGTH_FEATURE,
    LOCATION_FEATURE,
    MULTIPLE_LINK_DATA_DIR,
    N_COMPONENTS_MULTIPLE_LINKS,
    POWER_FEATURE,
    ROADM_SIDE_FEATURE,
    SEED,
    TEST_SIZE_RATIO,
)
from src.label_derivations import calculate_multiple_link_label_derivations
from src.load_dataset import Scenario, concat_helper, load_dataset
from src.preprocessing import (
    create_fit_transfrom_standard_scaler,
    custom_train_test_split,
    make_pipeline,
)
from src.project_init import get_logger
from src.result_exporter import (
    create_classification_report,
    create_compression_report,
    low_dim_embedding_plot,
)

export_path = pathlib.Path(EXP_DIR)


def col_name_alter_helper(data: pd.DataFrame) -> pd.DataFrame:
    data.columns = [N_COMPONENTS_MULTIPLE_LINKS, N_COMPONENTS_MULTIPLE_LINKS + 1]
    return data


def multiple_scenario_low_dim_plot(
    train_mat_features: pd.DataFrame,
    train_col_features: pd.DataFrame,
    test_mat_features: pd.DataFrame,
    test_col_features: pd.DataFrame,
    test_targets: pd.DataFrame,
) -> None:
    # exports the classification report of multiple link scenario
    export_path = pathlib.Path(EXP_DIR)
    train_features = concat_helper(train_mat_features.reset_index(drop=True), train_col_features)
    test_features = concat_helper(test_mat_features.reset_index(drop=True), test_col_features)
    test_targets_class = test_targets + 1  # target should be either 1 or 2 dBm
    low_dim_embedding_plot(
        train_features,
        test_features,
        test_targets_class,
        export_path / f'multiple_scenario.png',
        f'Multiple Links Scenario Embeddings',
    )
    low_dim_embedding_plot(
        train_features,
        test_features,
        test_targets_class,
        export_path / f'multiple_scenario_zoommed.png',
        f'Multiple Links Scenario Embeddings (Limited Axis)',
        limit_axis=True,
    )


def execute_multiple_links_scenario() -> None:
    # initialize the logger
    logger = get_logger(export_path / MULTIPLE_LINK_DATA_DIR)
    # load dataset
    features, labels = load_dataset(Scenario.MULTIPLE_LINK, seed=SEED)
    labels[[DISTANCE_FEATURE, ROADM_SIDE_FEATURE, POWER_FEATURE]] = calculate_multiple_link_label_derivations(
        labels[[LINK_LENGTH_FEATURE, LOCATION_FEATURE, POWER_FEATURE]]
    )
    # train-test split
    (
        feature_constellation_train,
        feature_constellation_test,
        feature_column_train,
        feature_column_test,
        target_train,
        target_test,
    ) = custom_train_test_split(
        features,
        labels[[DISTANCE_FEATURE, ROADM_SIDE_FEATURE]],
        labels[POWER_FEATURE],
        test_size=TEST_SIZE_RATIO,
        random_state=SEED,
    )
    # pre-compression preprocessing
    feature_constellation_scaler, scaled_train_constellation_features = create_fit_transfrom_standard_scaler(
        feature_constellation_train
    )
    feature_column_scaler, scaled_train_column_features = create_fit_transfrom_standard_scaler(
        feature_column_train, column_wise=True
    )

    logger.info('dataset is loaded and preprocessed with standard scaler and split into train-test')
    # low dimension insights
    multiple_scenario_low_dim_plot(
        scaled_train_constellation_features,
        scaled_train_column_features,
        feature_constellation_scaler.transform(feature_constellation_test),
        feature_column_scaler.transform(feature_column_test),
        target_test,
    )
    # compression
    pca = PCA(n_components=N_COMPONENTS_MULTIPLE_LINKS, random_state=SEED)
    train_embeddings = pca.fit_transform(scaled_train_constellation_features)
    # post-compression preprocessing
    embedding_scaler, scaled_train_embeddings = create_fit_transfrom_standard_scaler(
        pd.DataFrame(train_embeddings), column_wise=True
    )
    logger.info(f'features are compressed with {COMPRESSION_METHOD} and scaled again with another standard scaler')
    feature_constellation_compression_pipeline = make_pipeline([feature_constellation_scaler.transform, pca.transform])
    feature_constellation_compress_decompress_pipeline = make_pipeline(
        [
            feature_constellation_compression_pipeline,
            pca.inverse_transform,
            feature_constellation_scaler.inverse_transform,
        ]
    )
    create_compression_report(
        method=COMPRESSION_METHOD,
        original_dim=feature_constellation_train.shape[1],
        compressed_dim=N_COMPONENTS_MULTIPLE_LINKS,
        features=feature_constellation_test,
        compressor_fwd_path=feature_constellation_compress_decompress_pipeline,
        logger=logger,
    )
    # classifier train
    scaled_train_features = concat_helper(scaled_train_embeddings, col_name_alter_helper(scaled_train_column_features))
    svm_classifier = SVC().fit(scaled_train_features, target_train)
    feature_constellation_fwd_pipeline = make_pipeline(
        [feature_constellation_compression_pipeline, pd.DataFrame, embedding_scaler.transform]
    )
    scaled_test_column_features = feature_column_scaler.transform(feature_column_test)
    scaled_test_featuers = concat_helper(
        feature_constellation_fwd_pipeline(feature_constellation_test),
        col_name_alter_helper(scaled_test_column_features),
    )
    predictions = svm_classifier.predict(scaled_test_featuers)
    confusion_mat = confusion_matrix(target_test, predictions)
    # create classification report
    create_classification_report(
        'multiple links scenario', 'PCA + SVM', 'dBm', confusion_mat, predictions, target_test, logger
    )
