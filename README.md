#  Music Genre Classification Using Audio Data

This project aims to classify music tracks into various genres based on audio features extracted from raw sound files. We leverage signal processing, feature engineering, and machine learning/deep learning techniques to train accurate and robust classifiers.

---

##  Objective

To build a supervised machine learning model that can classify music into genres (e.g., Pop, Classical, Jazz, Rock) using time-series and frequency-based audio features.

---

##  Technologies Used

- **Python 3.x**
- **Librosa** – for audio signal processing
- **NumPy / Pandas** – for data manipulation
- **Matplotlib / Seaborn** – for visualization
- **Scikit-learn** – for traditional machine learning models
- **TensorFlow / Keras** – for deep learning (CNN)
- **Jupyter Notebooks** – for exploration and reporting

---

##  Methods Used

###  Feature Extraction
- **MFCC (Mel Frequency Cepstral Coefficients)**
- **Chroma Features**
- **Spectral Centroid**
- **Spectral Bandwidth**
- **Zero-Crossing Rate**
- **Tempo (BPM)**
- Extracted using `librosa` from raw `.wav` files.

###  Classification Models
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Convolutional Neural Network (CNN)** trained on spectrograms

###  Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

##  Key Findings

- **MFCCs and Chroma** were the most influential features for distinguishing between genres.
- **CNNs trained on spectrogram images** significantly outperformed classical models.
- **SVM** achieved reasonable accuracy but struggled with overlapping genres.
- **Genre distribution** imbalance affected prediction quality – mitigated using stratified sampling and data augmentation.
- Noise, tempo, and rhythm strongly influence model prediction confidence in rhythmic genres like pop and rock.

---

##  Files Included

- **Report**: Contains a detailed explanation of the approach, feature engineering, visualizations, models used, evaluation metrics, and conclusions drawn from the experiments.
- **Code**: Includes:
  - Audio preprocessing scripts
  - Feature extraction using Librosa
  - Model training and evaluation scripts
  - Visualization notebooks (confusion matrix, accuracy plots, etc.)
- **Visuals**: Includes genre distribution plots, confusion matrices, and training accuracy/loss graphs.
- **Dataset**: GTZAN or similar genre-labeled dataset used for training and evaluation.


---
##  Key Insights

- **Network topology** (e.g., dense vs. sparse feature space) impacts both the performance and generalizability of classification models.
- **MFCCs (Mel-Frequency Cepstral Coefficients)**, chroma features, and spectral contrast emerged as the most informative features.
- **CNNs** and deep learning models outperformed classical ML methods (e.g., SVM, KNN) in both accuracy and robustness.
- Dimensionality reduction and class balancing significantly improved classification consistency.
- Strategies like **data augmentation** and **genre-aware community detection** (in spectral clustering) enhanced genre separation in sparse feature space.

---

##  Future Work

- Incorporate more complex architectures like **Transformer-based models** or **CRNNs (Convolutional Recurrent Neural Networks)**.
- Extend classification to **multi-label genre tagging** (e.g., a song may be both "Jazz" and "Fusion").
- Add real-time classification interface for streaming or live audio input.
- Explore **user sentiment** and **lyrical content** alongside audio features to improve prediction confidence.
- Apply the pipeline to **multi-lingual or cross-cultural music datasets**.

---
