import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.constants import (
    COMPRESSION_METHOD,
    DISTANCE_FEATURE,
    DPI,
    EXP_DIR,
    N_COMPONENTS,
    SEED,
    SINGLE_LINK_DATA_DIR,
    SINGLE_LINK_MODE_FEATURE,
    SINGLE_LINK_SPAN_COUNT_FEATURE,
    TEST_SIZE_RATIO,
)
from src.label_derivations import calculate_single_link_lp_length
from src.load_dataset import Scenario, load_dataset
from src.preprocessing import (
    create_fit_transfrom_standard_scaler,
    custom_train_test_split,
    make_pipeline,
)
from src.project_init import get_logger
from src.result_exporter import (
    calculate_normalized_embeddings,
    create_compression_report,
    create_regression_report,
    grouping_helper_function,
    low_dim_embedding_plot,
)

export_path = pathlib.Path(EXP_DIR)


def single_scenario_low_dim_plot(
    train_features: pd.DataFrame, test_features: pd.DataFrame, test_targets: pd.DataFrame, modes: pd.DataFrame
) -> None:
    # exports the regression report of single link scenario
    embeddings = calculate_normalized_embeddings(train_features, test_features)
    modes_group = grouping_helper_function(modes)
    fig, ax = plt.subplots()
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(modes_group))))
    for val, grp in modes_group:
        indices = grp.index
        low_dim_embedding_plot(
            train_features.iloc[indices],
            test_features.iloc[indices],
            test_targets.iloc[indices],
            export_path / f'single_scenario({val}).png',
            f'Single Link Scenario Embeddings ({val} Mode)',
        )
        grp_embeddings = embeddings[indices]
        cluster_center = grp_embeddings.mean(axis=0)
        color = next(colors)
        ax.scatter(*grp_embeddings.T, s=1, c=color)
        ax.text(*cluster_center, str(val), color=color)
    ax.set_xlabel('1st Scaled Principle Component')
    ax.set_ylabel('2nd Scaled Principle Component')
    ax.set_title('Single Link Scenario Embeddings')
    fig.savefig(export_path / 'single_scenario.png', dpi=DPI)


def execute_single_link_scenario() -> None:
    # initialize the logger
    logger = get_logger(export_path / SINGLE_LINK_DATA_DIR)
    # load dataset
    features, labels = load_dataset(Scenario.SINGLE_LINK, seed=SEED)
    labels[DISTANCE_FEATURE] = calculate_single_link_lp_length(
        labels[[SINGLE_LINK_SPAN_COUNT_FEATURE, SINGLE_LINK_MODE_FEATURE]]
    )
    # train-test split
    feature_train, feature_test, target_train, target_test, _, mode_test = custom_train_test_split(
        features,
        labels[DISTANCE_FEATURE],
        labels[SINGLE_LINK_MODE_FEATURE],
        test_size=TEST_SIZE_RATIO,
        random_state=SEED,
    )
    # pre-compression preprocessing
    label_scaler, scaled_train_labels = create_fit_transfrom_standard_scaler(target_train)
    feature_scaler, scaled_train_features = create_fit_transfrom_standard_scaler(feature_train)
    logger.info('dataset is loaded and preprocessed with standard scaler and split into train-test')
    # low dimension insights
    single_scenario_low_dim_plot(scaled_train_features, feature_scaler.transform(feature_test), target_test, mode_test)
    # compression
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    train_embeddings = pca.fit_transform(scaled_train_features)
    # post-compression preprocessing
    embedding_scaler, scaled_train_embeddings = create_fit_transfrom_standard_scaler(train_embeddings)
    feature_fwd_pipeline = make_pipeline([feature_scaler.transform, pca.transform, embedding_scaler.transform])
    logger.info(f'features are compressed with {COMPRESSION_METHOD} and scaled again with another standard scaler')
    create_compression_report(
        method=COMPRESSION_METHOD,
        original_dim=features.shape[1],
        compressed_dim=N_COMPONENTS,
        features=feature_test,
        compressor_fwd_path=make_pipeline([feature_fwd_pipeline, pca.inverse_transform]),
        logger=logger,
    )
    # regressor train
    linear_regressor = LinearRegression().fit(scaled_train_embeddings, scaled_train_labels)
    r2_test_score = linear_regressor.score(feature_fwd_pipeline(feature_test), label_scaler.transform(target_test))
    single_link_e2e_pipeline = make_pipeline([feature_fwd_pipeline, linear_regressor.predict])
    # create regression report
    create_regression_report(
        'single link scenario',
        'PCA + LinearRegression',
        'km',
        r2_test_score,
        feature_test,
        target_test,
        mode_test,
        single_link_e2e_pipeline,
        label_scaler,
        logger=logger,
    )
