import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.constants import (
    COMPRESSION_METHOD,
    DISTANCE_FEATURE,
    LINK_LENGTH_FEATURE,
    LOCATION_FEATURE,
    N_COMPONENTS,
    POWER_FEATURE,
    ROADM_SIDE_FEATURE,
    SEED,
    SINGLE_LINK_MODE_FEATURE,
    SINGLE_LINK_SPAN_COUNT_FEATURE,
    TEST_SIZE_RATIO,
)
from src.label_derivations import (
    calculate_multiple_link_label_derivations,
    calculate_single_link_lp_length,
)
from src.load_dataset import Scenario, load_dataset
from src.preprocessing import (
    create_fit_transfrom_standard_scaler,
    make_pipeline,
    remove_column_features,
)
from src.result_exporter import (
    create_compression_report,
    create_regression_report,
    print_announcement,
)


def main():
    # '''single link scenario'''
    # print_announcement('single link scenario: lightpath distance prediction with constellation samples')
    # # load dataset
    # single_link_labels = [DISTANCE_FEATURE]
    # features, labels = load_dataset(Scenario.SINGLE_LINK, seed=SEED)
    # labels[DISTANCE_FEATURE] = calculate_single_link_lp_length(
    #     labels[[SINGLE_LINK_SPAN_COUNT_FEATURE, SINGLE_LINK_MODE_FEATURE]]
    # )
    # # train-test split
    # feature_train, feature_test, label_train, label_test = train_test_split(
    #     features, labels[single_link_labels], test_size=TEST_SIZE_RATIO, random_state=SEED
    # )
    # # pre-compression preprocessing
    # label_scaler, scaled_train_labels = create_fit_transfrom_standard_scaler(label_train)
    # feature_scaler, scaled_train_features = create_fit_transfrom_standard_scaler(feature_train)
    # print('dataset is loaded and preprocessed with standard scaler and split into train-test')
    # # compression
    # pca = PCA(n_components=N_COMPONENTS)
    # train_embeddings = pca.fit_transform(scaled_train_features)
    # # post-compression preprocessing
    # embedding_scaler, scaled_train_embeddings = create_fit_transfrom_standard_scaler(train_embeddings)
    # feature_fwd_pipeline = make_pipeline([feature_scaler.transform, pca.transform, embedding_scaler.transform])
    # print(f'features are compressed with {COMPRESSION_METHOD} and scaled again with another standard scaler')
    # create_compression_report(
    #     COMPRESSION_METHOD,
    #     features.shape[1],
    #     N_COMPONENTS,
    #     feature_test,
    #     make_pipeline([feature_fwd_pipeline, pca.inverse_transform]),
    #     pca,
    # )
    # # regressor train
    # linear_regressor = LinearRegression().fit(scaled_train_embeddings, scaled_train_labels)
    # r2_test_score = linear_regressor.score(feature_fwd_pipeline(feature_test), label_scaler.transform(label_test))
    # single_link_e2e_pipeline = make_pipeline([feature_fwd_pipeline, linear_regressor.predict])
    # create_regression_report(
    #     'single link scenario',
    #     'PCA + LinearRegression',
    #     'km',
    #     r2_test_score,
    #     feature_test,
    #     label_test,
    #     single_link_e2e_pipeline,
    #     label_scaler,
    # )

    '''multiple link scenario'''
    print_announcement(
        'multiple links scenario: launch power prediction with constellation samples and sample location'
    )
    # load dataset
    features, labels = load_dataset(Scenario.MULTIPLE_LINK, seed=SEED)
    labels[[DISTANCE_FEATURE, ROADM_SIDE_FEATURE, POWER_FEATURE]] = calculate_multiple_link_label_derivations(
        labels[[LINK_LENGTH_FEATURE, LOCATION_FEATURE, POWER_FEATURE]]
    )
    # train-test split
    augmented_features = pd.concat([features, labels[[DISTANCE_FEATURE, ROADM_SIDE_FEATURE]]], axis=1)
    feature_train, feature_test, label_train, label_test = train_test_split(
        augmented_features, labels[POWER_FEATURE], test_size=TEST_SIZE_RATIO, random_state=SEED
    )
    # pre-compression preprocessing
    label_scaler, scaled_train_labels = create_fit_transfrom_standard_scaler(label_train)
    feature_scaler, scaled_train_features = create_fit_transfrom_standard_scaler(
        feature_train, column_features=[DISTANCE_FEATURE, ROADM_SIDE_FEATURE]
    )
    print('dataset is loaded and preprocessed with standard scaler and split into train-test')
    # compression
    pca = PCA(n_components=N_COMPONENTS)
    train_embeddings = pca.fit_transform(remove_column_features(scaled_train_features))
    # post-compression preprocessing
    embedding_scaler, scaled_train_embeddings = create_fit_transfrom_standard_scaler(pd.DataFrame(train_embeddings))
    feature_fwd_pipeline = make_pipeline(
        [feature_scaler.transform, remove_column_features, pca.transform, pd.DataFrame, embedding_scaler.transform]
    )
    print(f'features are compressed with {COMPRESSION_METHOD} and scaled again with another standard scaler')
    create_compression_report(
        COMPRESSION_METHOD,
        features.shape[1],
        N_COMPONENTS,
        feature_test,
        make_pipeline([feature_fwd_pipeline, pca.inverse_transform]),
        pca,
    )
    # regressor train
    linear_regressor = LinearRegression().fit(scaled_train_embeddings, scaled_train_labels)
    r2_test_score = linear_regressor.score(feature_fwd_pipeline(feature_test), label_scaler.transform(label_test))
    single_link_e2e_pipeline = make_pipeline([feature_fwd_pipeline, linear_regressor.predict])
    create_regression_report(
        'multiple links scenario',
        'PCA + LinearRegression',
        'dBm',
        r2_test_score,
        feature_test,
        label_test,
        single_link_e2e_pipeline,
        label_scaler,
    )

    print_announcement('All done!')


if __name__ == '__main__':
    main()
