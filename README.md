# OmniAnomaly



### Anomaly Detection for Multivariate Time Series through Modeling Temporal Dependence of Stochastic Variables

OmniAnomaly is a stochastic recurrent neural network model which glues Gated Recurrent Unit (GRU) and Variational auto-encoder (VAE), its core idea is to learn the normal patterns of multivariate time series and uses the reconstruction probability to do anomaly judgment. 



## Getting Started

#### Clone the repo

```
git clone https://github.com/smallcowbaby/OmniAnomaly && cd OmniAnomaly
```

#### Get data

SMD (Server Machine Dataset) is in folder `ServerMachineDataset`. 

You can get the public datasets (SMAP and MSL) using:

```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

#### Install dependencies (with python 3.5, 3.6) 

(virtualenv is recommended)

```shell
pip install -r requirements.txt
```

#### Preprocess the data

```shell
python data_preprocess.py <dataset>
```

where `<dataset>` is one of `SMAP`, `MSL` or `SMD`.

#### Run the code

```
python main.py
```

If you want to change the default configuration, you can edit `ExpConfig` in `main.py` or overwrite the config in `main.py` using command line args. For example:

```
python main.py --dataset='MSL' --max_epoch=20
```



## Data

### Dataset Information

| Dataset name| Number of entities | Number of dimensions | Training set size |Testing set size |Anomaly ratio(%)|
|------|----|----|--------|--------|-------|
| SMAP | 55 | 25 | 135183 | 427617 | 13.13 |
|MSL | 27 | 55 | 58317 | 73729 | 10.72|
|SMD | 28 |38 | 708405 | 708420 | 4.16 |



### SMAP and MSL

SMAP (Soil Moisture Active Passive satellite) and MSL (Mars Science Laboratory rover) are two public datasets from NASA.

For more details, see: <https://github.com/khundman/telemanom>



### SMD

SMD (Server Machine Dataset) is a new 5-week-long dataset. We collected it from a large Internet company. This dataset contains 3 groups of entities. Each of them is named by `machine-<group_index>-<index>`.

SMD is made up by data from 28 different machines, and the 28 subsets should be trained and tested separately. For each of these subsets, we divide it into two parts of equal length for training and testing. We provide labels for whether a point is an anomaly and the dimensions contribute to every anomaly.

Thus SMD is made up by the following parts:

* train: The former half part of the dataset.
* test: The latter half part of the dataset.
* test_label: The label of the test set. It denotes whether a point is an anomaly. 
* interpretation_label: The lists of dimensions contribute to each anomaly.

concatenate



## Processing

With the default configuration, `main.py` follows these steps:

* Train the model with training set, and validate at a fixed frequency. Early stop method is applied by default.
* Test the model on both training set and testing set, and save anomaly score in `train_score.pkl` and `test_score.pkl`.
* Find the best F1 score on the testing set, and print the results.
* Init POT model on `train_score` to find the threshold of anomaly score, and using this threshold to predict on the testing set.


## Training loss

The figure below are the training loss of our model on MSL and SMAP, which indicates that our model can converge well on these two datasets.

![image](https://github.com/smallcowbaby/OmniAnomaly/blob/master/images/MSL_loss.png)
![image](https://github.com/smallcowbaby/OmniAnomaly/blob/master/images/SMAP_loss.png)

