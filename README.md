# Credit Card Approval Prediction

## Project Descirption
Our project, ”Credit Card Approval Prediction”, aims to revolutionize the credit approval process by developing a predictive model that enhances accuracy, fairness, and efficiency. We will leverage a diverse dataset to train and test our models while implementing fairness-aware techniques. This project, outlined in this proposal, provides a comprehensive methodology, evaluation methods, and a GitHub repository, fostering collaboration and progress tracking. With a team of dedicated members, we are committed to making regular commits and ensuring transparency throughout the project. The successful execution of this project has the potential to significantly impact individuals seeking credit and the financial institutions making lending decisions.

## Team Members
- Yusuf Faruk Güldemir - 150210302
- Hasan Taha Bağcı - 150210338

## Dataset
Two datasets used in this project are:
- [Application Record](https://www.kaggle.com/rikdifos/credit-card-approval-prediction) with 438557 rows and 18 columns
- [Credit Record](https://www.kaggle.com/rikdifos/credit-card-approval-prediction) with 1048575 rows and 3 columns
- `processed_data.csv` our final dataset.

## Project Structure
```bash
│
├── README.md
│
├── data
│   ├── application_record.csv
│   ├── credit_record.csv
│   └── processed_data.csv
│
├── docs
│   ├── Proposal.pdf
│   └── Intermediate Report.pdf
│
├── utils
│   ├── datapreprocess.py
│   ├── model.py
│   └── plotdata.py
│    
├── eploration.ipynb
│
├── model.ipynb
│
├── Progress.txt   
│
└── requirements.txt
```

## Installation
Required packages:
- Python3
- jupyter
- imblearn
- pandas==1.4.4
- scikit-learn==1.0.2
- seaborn==0.11.2
- xgboost==1.7.5
- matplotlib==3.5.2

To install required packages run `pip install -r requirements.txt` in the project directory.

## Usage
To run the project, run `jupyter notebook` in the project directory and open `exploration.ipynb` and `model.ipynb` files.
You can run the cells in the exploration notebook to see the data exploration process and create the `processed_data.csv` file. 
After that, you can run the cells in the model notebook to see the model training process and the results, anlysis and evaluation of the models.
