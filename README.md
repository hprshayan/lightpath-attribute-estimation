# Lightpath Attribute Estimation
This repository includes the code for lightpath length and launch power esitmation with machine learning.

## Code Execution
**The code requires at least Python 3.10 for execution**

Create a virtual environment and install the requirements
```sh
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```
After the environment setup finished, the code could be executed by:
```sh
python main.py
```

## Code Structure
The structure of the code is:
```sh
main.py
README.md
requirements.txt
src
├── compressor.py
├── constants.py
├── __init__.py
├── label_derivations.py
├── load_dataset.py
├── multiple_links_sc_classification.py
├── preprocessing.py
├── project_init.py
├── result_exporter.py
└── single_link_sc_regression.py
data
├── accessible_dataset
│   ├── 16QAM-multipleLink-s1_data.zip
│   ├── 16QAM-multipleLink-s1_description.pdf
│   ├── 16QAM-singleLink-s1_data.zip
│   ├── 16QAM-singleLink-s1_description.pdf
│   ├── MANIFEST.TXT
│   ├── multiple_link_scenario
│   |   └── csv files ....
│   ├── Readme.txt
│   └── single_link_scenario
│       ├── degradation
│       |   └── csv files ....
│       ├── optimal
│       |   └── csv files ....
│       ├── sub-optimal
│       |   └── csv files ....
└── constellation-dataset.zip
exports
├── multiple_link_scenario.log
├── multiple_scenario.png
├── multiple_scenario_zoommed.png
├── single_link_scenario.log
├── single_scenario(degradation).png
├── single_scenario(optimal).png
├── single_scenario.png
└── single_scenario(sub-optimal).png
report
├── figures
│   ├── models.png
│   ├── multiple_scenario_zoommed.png
│   ├── single_scenario(optimal).png
│   └── single_scenario.png
├── IEEEtran.cls
├── manuscript.pdf
└── manuscript.tex
```

## Results
The images and log files are saved in the `exports` directory.

Also, you can find experiments report in the `report/manuscript.pdf` path.

The output of the terminal while the code is being executed is shown below. The output logs are also saved in `exports/*.log` files.
```
initializing the demo...
demo project is initialized

######################################################################################################
########### single link scenario: lightpath distance prediction with constellation samples ###########
######################################################################################################
dataset is loaded and preprocessed with standard scaler and split into train-test
features are compressed with PCA and scaled again with another standard scaler
compression with PCA method is done.
compression rate: 98.78%	reconstruction MAE: 0.0475	reconstruction MAPE: 3.50%
compressed data dimension: 50
single link scenario regression with PCA + LinearRegression approach is done.
test score (coefficeint of determination): 0.99916
here are some predictions (with km as unit):
  index          Mode             prediction          target          |target-prediction|
  593            optimal          720.4               720.0           0.4
  141            optimal          158.9               160.0           1.1
  1670           degradation      1146.2              1160.0          13.8
  192            optimal          965.9               960.0           5.9
  1684           degradation      1151.6              1160.0          8.4
  1622           degradation      1069.4              1080.0          10.6
  2103           sub-optimal      1335.6              1340.0          4.4
  1497           degradation      1009.4              1000.0          9.4
  1255           degradation      1889.2              1880.0          9.2
  356            optimal          1428.0              1440.0          12.0

######################################################################################################
## multiple links scenario: launch power prediction with constellation samples and sample location ###
######################################################################################################
dataset is loaded and preprocessed with standard scaler and split into train-test
features are compressed with PCA and scaled again with another standard scaler
compression with PCA method is done.
compression rate: 99.69%	reconstruction MAE: 1.1959	reconstruction MAPE: 67.22%
compressed data dimension: 50
multiple links scenario classification with PCA + SVM approach is done.
confusion matrix:
                1 dBm (prediction)  2 dBm (prediction)
1 dBm (target)                 132                   0
2 dBm (target)                  34                  98
accuracy: 0.87121
here are some predictions (with dBm as unit):
  index          prediction          target
  501            1                   1              
  141            1                   1              
  496            1                   1              
  578            2                   2              
  463            1                   1              
  56             1                   1              
  35             1                   2              
  771            1                   1              
  63             2                   2              
  285            1                   2              

######################################################################################################
###################################### All done in 73 seconds! #######################################
######################################################################################################
```

## Credits
The dataset used in this repository is downloaded from [1].
The motivation of the use cases is partially credited to [2].

[1] Ruiz Ramı́rez, M., Velasco Esteban, L. & Sequeira, D. Optical Constellation Analysis (OCATA). (CORA.Repositori de Dades de Recerca,2022), https://doi.org/10.34810/data146

[2] Ruiz, M., Sequeira, D. & Velasco, L. Deep learning-based real-time analysis of lightpath optical constellations [Invited]. Journal Of Optical Communications And Networking. 14, C70-C81 (2022)