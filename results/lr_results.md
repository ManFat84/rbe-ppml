# Logistic regression results

Settings: {'test_size': 0.2, 'split_seed': 42, 'epochs': 80, 'batch_size': 64, 'learning_rate': 0.2, 'shuffle_seed': 0, 'sigmoid_degree': 3, 'sigmoid_range': 8.0}

## Table III (metrics on the test set)

| Dataset | (#N, #F) | #Iter | Unenc. ACC | Unenc. AUC | Unenc. MSE | Poly ACC | Poly AUC | Poly MSE | Enc. ACC | Enc. AUC | Enc. MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | (569, 30) | 80 | 93.86% | 99.90% | 0.048594 | 93.86% | 99.83% | 0.057510 | 93.86% | 99.83% | 0.057510 |
| heart_disease | (303, 13) | 80 | 85.25% | 92.86% | 0.111970 | 81.97% | 92.42% | 0.116532 | 81.97% | 92.42% | 0.116532 |

## Table II (runtime)

| Metric (s) | breast_cancer | heart_disease |
|---|---|---|
| baseline training time | 0.04 | 0.01 |
| dataset encryption time | 14.63 | 3.44 |
| encrypted training time (cloud) | 44.09 | 8.88 |
| refresh time (key owner) | 21.66 | 4.71 |
| encrypted training time (total) | 66.01 | 13.66 |
| encrypted inference, test set | 0.07 | 0.02 |
| key generation + decryption | 0.05 | 0.03 |
| total encrypted pipeline | 80.75 | 17.15 |
| encryption overhead | 80.71 | 17.14 |

## Consistency checks

| Dataset | SGD updates | refreshes | max scale exponent | max |param diff| enc. vs poly | prediction agreement |
|---|---|---|---|---|---|
| breast_cancer | 640 | 19840 | 11 | 1.69e-04 | 100.00% |
| heart_disease | 320 | 4480 | 11 | 1.21e-04 | 100.00% |
