# Simultaneous FX Price Prediction by Using Temporal Multivariate Graph Neural Networks

## Requirements and Setup

* **StemGNN**: ([instruction to setup StemGNN](https://github.com/microsoft/StemGNN)).

## Abstract

In the graduation project, we try to make simultaneous Forex price prediction by using temporal multivariate graph neural networks. Models such as LSTM used for Forex forecasting today only deal with historical data for a currency, rather than looking at historical data of other currencies. Therefore, they are insufficient. For this, we have worked on two existing algorithms and developed them. One of them is the model called StemGNN, which was previously used to estimate the number of Covid-19 cases. The model includes a new approach that combines the intra-series temporal patterns and inter-series correlations jointly and it is used for the first time in the field of finance. In other words, when estimating the price of a currency, StemGNN first examines the historical data of that currency and then the historical data of other currencies. The other model is called [DeepGLO](https://github.com/rajatsen91/deepglo), which uses convolutional neural networks. Its results are not as bright as StemGNN. However, it still provides closer predictions than StemGNN for highly volatile currencies such as the Russian Ruble and Indian Rupee. Therefore, only StemGNN will be showed on this repository. Finally, we created a portfolio of our results. We tried to maximize the return per risk, the sharpe ratio, and we achieved satisfying results.

## Results

While evaluating our results, we took the mean absolute percentage error (MAPE) as a basis and compared the overall MAPE values of LSTM and the model we developed, StemGNN. LSTM have a high test MAPE which is 5.7%. On the other hand, when we used StemGNN we reached 2%, which is an extremely low rate for MAPE, and we got the best result.


**Some Chart Examples for Predict vs Target Values**
(JPY: Japanese Yen, AUD: Australian Dollar, CNY: Chinese Yuan)

![1- EURO](https://user-images.githubusercontent.com/63553314/190496662-03b10b24-2bf8-403d-92ed-d5ce3d7e7b5f.png)
![3 - JPY](https://user-images.githubusercontent.com/63553314/190497367-76b671dc-7466-42e0-81fb-d8f5a95e875c.png)
![5- AUD](https://user-images.githubusercontent.com/63553314/190497830-3f8a37d7-d581-4c23-a918-a90b3a6c3382.png)
![8 - CNY](https://user-images.githubusercontent.com/63553314/190497928-f63b4e65-964a-48e8-96a2-f3fa2f8f84ac.png)

**Table 1** Mape Results of LSTM vs StemGNN
| Currencies | Mape of LSTM | Mape of StemGNN |
| ------------- | ------------- | ------------- |
| EUR | 1.076 | 0.617 |
| CAD | 2.154 | 0.972 |
| JPY | 3.033 | 1.052 |
| GBP | 1.818 | 0.784 |
| AUD | 3.437 | 0.832 |
| RUB | 24.85 | 3.447 |
| CHF | 4.270 | 1.065 |
| CNY | 1.566 | 0.410 |
| INR | 9.893 | 3.807 |

**Table 2** Weight Ratio of StemGNN in the Portfolio

| Currencies | Mape of StemGNN |
| ------------- | ------------- |
| EUR | 10.42% |
| CAD | 0.86% |
| JPY | 0.00% |
| GBP | 0.03% |
| AUD | 2.79% |
| RUB | 0.00% |
| CHF | 46.07% |
| CNY | 34.72% |
| INR | 5.11% |

**Table 3** Portfolio Results of DeepGLO vs StemGNN

| Results | DeepGLO| StemGNN |
| ------------- | ------------- | ------------- |
| Total Return | 1.10% | 2.00% |
| Volatility | 1.20% | 0.70% |
| Sharpe Ratio | 0.78 | **2.63** |

