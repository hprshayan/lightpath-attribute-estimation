import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from src.constants import (
    COMPRESSION_METHOD,
    DISTANCE_FEATURE,
    LINK_LENGTH_FEATURE,
    LOCATION_FEATURE,
    N_COMPONENTS,
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
from src.result_exporter import create_classification_report, create_compression_report


def execute_multiple_links_scenario() -> None:
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

    print('dataset is loaded and preprocessed with standard scaler and split into train-test')
    # compression
    pca = PCA(n_components=N_COMPONENTS)
    train_embeddings = pca.fit_transform(scaled_train_constellation_features)
    # post-compression preprocessing
    embedding_scaler, scaled_train_embeddings = create_fit_transfrom_standard_scaler(pd.DataFrame(train_embeddings))
    print(f'features are compressed with {COMPRESSION_METHOD} and scaled again with another standard scaler')
    feature_constellation_compression_pipeline = make_pipeline([feature_constellation_scaler.transform, pca.transform])
    feature_constellation_compress_decompress_pipeline = make_pipeline(
        [
            feature_constellation_compression_pipeline,
            pca.inverse_transform,
            feature_constellation_scaler.inv_transform,
        ]
    )
    create_compression_report(
        method=COMPRESSION_METHOD,
        original_dim=feature_constellation_train.shape[1],
        compressed_dim=N_COMPONENTS,
        features=feature_constellation_test,
        compressor_fwd_path=feature_constellation_compress_decompress_pipeline,
        compressor_explainer=pca,
    )
    # classifier train
    scaled_train_features = concat_helper(scaled_train_embeddings, pd.DataFrame(scaled_train_column_features))
    svm_classifier = SVC().fit(scaled_train_features, target_train)
    feature_constellation_fwd_pipeline = make_pipeline(
        [feature_constellation_compression_pipeline, embedding_scaler.transform]
    )
    scaled_test_column_features = feature_column_scaler.transform(feature_column_test)
    scaled_test_featuers = concat_helper(
        feature_constellation_fwd_pipeline(feature_constellation_test), scaled_test_column_features
    )
    predictions = svm_classifier.predict(scaled_test_featuers)
    confusion_mat = confusion_matrix(target_test, predictions)
    create_classification_report('multiple links scenario', 'PCA + SVM', 'dBm', confusion_mat, predictions, target_test)
