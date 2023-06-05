from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.constants import (
    COMPRESSION_METHOD,
    DISTANCE_FEATURE,
    N_COMPONENTS,
    SEED,
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
from src.result_exporter import (
    create_compression_report,
    create_regression_report,
    single_scenario_low_dim_plot,
)


def execute_single_link_scenario() -> None:
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
    print('dataset is loaded and preprocessed with standard scaler and split into train-test')
    # low dimension insights
    single_scenario_low_dim_plot(scaled_train_features, feature_scaler.transform(feature_test), target_test, mode_test)
    # compression
    pca = PCA(n_components=N_COMPONENTS)
    train_embeddings = pca.fit_transform(scaled_train_features)
    # post-compression preprocessing
    embedding_scaler, scaled_train_embeddings = create_fit_transfrom_standard_scaler(train_embeddings)
    feature_fwd_pipeline = make_pipeline([feature_scaler.transform, pca.transform, embedding_scaler.transform])
    print(f'features are compressed with {COMPRESSION_METHOD} and scaled again with another standard scaler')
    create_compression_report(
        method=COMPRESSION_METHOD,
        original_dim=features.shape[1],
        compressed_dim=N_COMPONENTS,
        features=feature_test,
        compressor_fwd_path=make_pipeline([feature_fwd_pipeline, pca.inverse_transform]),
        compressor_explainer=pca,
    )
    # regressor train
    linear_regressor = LinearRegression().fit(scaled_train_embeddings, scaled_train_labels)
    r2_test_score = linear_regressor.score(feature_fwd_pipeline(feature_test), label_scaler.transform(target_test))
    single_link_e2e_pipeline = make_pipeline([feature_fwd_pipeline, linear_regressor.predict])
    create_regression_report(
        'single link scenario',
        'PCA + LinearRegression',
        'km',
        r2_test_score,
        feature_test,
        target_test,
        single_link_e2e_pipeline,
        label_scaler,
    )
