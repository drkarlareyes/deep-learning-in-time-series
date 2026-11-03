# Deep learning for human activity recognition from IMU and motion-capture data

Notebooks that classify three activities (walking, sitting and squatting) from raw tri-axial signals with deep learning. The same pipeline runs on inertial (IMU) data and on optical motion-capture (Mocap) data, and compares four architectures: a dense network (DNN), a 1D CNN, an LSTM and a CNN-LSTM.

## Pipeline

1. **Input.** Six channels per sample (x, y and z from two sensors) and an activity label.
2. **Sliding windows** of 100 consecutive samples, moved one sample at a time.
3. **Models** (TensorFlow/Keras):
   - DNN: three dense layers of 32 units.
   - 1D CNN: two convolution layers (kernel 3) and a dense classifier.
   - LSTM: one LSTM layer of 100 units and a dense classifier.
   - CNN-LSTM: two convolution layers on sub-sequences of the window, then an LSTM of 100 units.
4. **Evaluation.** Accuracy, weighted precision, recall and F1, confusion matrices, and prediction time per window.

## Notebooks

| Notebook | Data | Model | Test data | Accuracy |
|---|---|---|---|---|
| `imu_cnn_lstm.ipynb` | IMU | CNN-LSTM | random 20% of windows | 99.96% |
| `imu_lstm.ipynb` | IMU | LSTM | random 20% of windows | 99.58% |
| `imu_lstm_8people.ipynb` | IMU, 8 people | LSTM | 2 other people | 91.9% |
| `mocap_dnn_8people.ipynb` | Mocap, 8 people | DNN | 2 other people | 87.0% |
| `mocap_cnn_8people.ipynb` | Mocap, 8 people | 1D CNN | 2 other people | 86.1% |
| `mocap_cnn_lstm_8people.ipynb` | Mocap, 8 people | CNN-LSTM | 2 other people | 88.4% |

Prediction time per window on the Mocap test data was about 0.06 ms (CNN), 0.10 ms (DNN) and 0.23 ms (CNN-LSTM).

**Note on the evaluation.** Consecutive windows overlap by 99 samples. In the random-split notebooks, neighbouring windows from the same recording fall into both the training and the test set, so those accuracies are optimistic. The 8-people notebooks train on eight people and test on two others, which gives the fairer estimate for new users.

## Data

The recordings are not included. The notebooks were written for Google Colab and read CSV files with six signal columns (`x1, y1, z1, x2, y2, z2`) and an activity label; the Mocap files also have a trial number. Change `path` in the loading cell to point to your files.

## Requirements

Python 3 with TensorFlow 2, NumPy, pandas, scikit-learn, matplotlib, seaborn and mlxtend.

## Related publication

Jaén-Vargas M, Reyes-Leiva KM, Fernandes F, Serrano Olmedo JJ. Effects of sliding window variation in the performance of acceleration-based human activity recognition using deep learning models. *PeerJ Computer Science* 8:e1052 (2022). https://doi.org/10.7717/peerj-cs.1052

## Author

Karla Miriam Reyes Leiva, PhD. [Website](https://drkarlareyes.github.io) · ORCID [0000-0002-3269-0830](https://orcid.org/0000-0002-3269-0830)

Code: MIT licence.
