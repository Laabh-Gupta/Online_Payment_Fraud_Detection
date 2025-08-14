# Online Payment Fraud Detection

![Fraud Detection](https://i.imgur.com/A8PA1wB.jpeg)

This project is a machine learning model to detect fraudulent online payment transactions. By analyzing transactional data, the model can predict with a high degree of accuracy whether a transaction is legitimate or fraudulent. This helps in preventing financial loss and securing online payment systems.

***

## 📜 Table of Contents
* [About the Project](#-about-the-project)
* [Key Features](#-key-features)
* [Dataset](#-dataset)
* [Technologies Used](#-technologies-used)
* [Getting Started](#-getting-started)
  * [Installation](#installation)
* [Usage](#-usage)
* [Model Training and Results](#-model-training-and-results)
* [Contributing](#-contributing)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

***

## 📖 About the Project

Online payment fraud is a significant and growing problem. This project aims to build a machine learning model that can effectively identify and flag fraudulent transactions in real-time. The model is trained on a large dataset of transactional data and uses a variety of features to distinguish between legitimate and fraudulent behavior. This is a classification problem, and this project explores using a **Random Forest Classifier** to solve it.

***

## ✨ Key Features

* **Fraud Detection:** Identifies fraudulent transactions with high accuracy.
* **Data Analysis:** Includes exploratory data analysis (EDA) to understand the patterns in the data.
* **Feature Engineering:** Selects and creates relevant features for the model.
* **Model Evaluation:** Uses various metrics to evaluate the performance of the model.

***

## 📊 Dataset

The model is trained on a synthetic dataset from Kaggle that mimics real-world transaction data. The dataset contains the following features:

* **step:** Represents a unit of time where 1 step equals 1 hour.
* **type:** The type of online transaction (e.g., `CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT`).
* **amount:** The amount of the transaction.
* **nameOrig:** The customer who started the transaction.
* **oldbalanceOrg:** The balance of the origin account before the transaction.
* **newbalanceOrig:** The balance of the origin account after the transaction.
* **nameDest:** The recipient of the transaction.
* **oldbalanceDest:** The balance of the destination account before the transaction.
* **newbalanceDest:** The balance of the destination account after the transaction.
* **isFraud:** A binary label indicating if the transaction is fraudulent (1) or not (0).

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection).

***

## 💻 Technologies Used

* **Python:** The primary programming language used.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For building and evaluating the machine learning model.
* **Matplotlib & Seaborn:** For data visualization.
* **Jupyter Notebook:** For interactive development and analysis.

***

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Installation

1.  Clone the repo
    ```sh
    git clone [https://github.com/Laabh-Gupta/Online_Payment_Fraud_Detection.git](https://github.com/Laabh-Gupta/Online_Payment_Fraud_Detection.git)
    ```
2.  Navigate to the project directory
    ```sh
    cd Online_Payment_Fraud_Detection
    ```

***

## Usage

You can run the project by opening the `Online_Payment_Fraud_Detection.ipynb` Jupyter Notebook file. The notebook contains all the steps from data loading and preprocessing to model training and evaluation.

To run the notebook:
```sh
jupyter notebook
***

## 📈 Model Training and Results

The model was trained using a Random Forest Classifier. Its performance is evaluated using metrics like Accuracy, Precision, Recall, and F1-Score. A confusion matrix is also used to visualize the performance of the model's classifications.

***

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

***

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

***

## 🙏 Acknowledgments

* [Kaggle](https://www.kaggle.com/) for the dataset.
* The open-source community for the amazing tools and libraries.
* [Project 03: Online Payment Fraud Detection Using Machine Learning](https://www.youtube.com/watch?v=C3fr3UMgLDo)