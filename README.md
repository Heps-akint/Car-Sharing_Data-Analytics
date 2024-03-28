# Car Sharing Data Analysis Project
This project revolves around the manipulation, management, and analysis of data from a car-sharing company. The dataset spans from January 2017 to August 2018, providing a comprehensive view of the customers' demand rate for car rentals on an hourly basis. The project is divided into two primary tasks: Database Management (Part I) and Data Analytics (Part II), each addressing a unique aspect of data science application.

## Overview
The goal of this project is twofold:

1. **Database Management**: Establish a structured database to efficiently store and manage the raw data, employing SQLite for database creation and management. This part includes data importation, categorisation, table modifications, and data extraction based on specific queries.

2. **Data Analytics**: Perform a thorough analysis of the data to uncover patterns, test hypotheses, and predict future trends using various statistical and machine learning methods.

## Dataset Description
The dataset, named **'CarSharing.csv'**, contains information about car rental demand, including timestamps, weather conditions, temperature, humidity, and wind speed, along with the demand rate for car rentals.

## Implementation Details
### Part I: Database Management
Implemented using Python and SQLite, the database management component involves:

- Creating and managing a SQLite database named **'CarSharingDB.db'**.
- Importing data from **'CarSharing.csv'** into the database.
- Categorizing data based on temperature to add a new column **'temp_category'**.
- Creating separate tables for temperature, weather conditions, and timestamp details for more efficient data management.

### Part II: Data Analytics
The data analytics component is implemented in Python, utilizing libraries such as Pandas, Matplotlib, SciPy, StatsModels, and Scikit-Learn. Key activities include:

- Data preprocessing (removing duplicates, handling null values).
- Testing hypotheses to find significant relationships between variables.
- Identifying seasonal or cyclic patterns in the data.
- Predicting future demand rates using ARIMA, Random Forest, and Deep Neural Network models.
- Classification of demand rates into categories and predicting them using Logistic Regression, Random Forest, and SVM.
- Clustering temperature data to find uniform clusters.

## Relevance to Data Science
This project exemplifies the end-to-end process of data science, from data management and preprocessing to sophisticated statistical analysis and predictive modeling. It highlights the importance of database management in handling large datasets and demonstrates the application of various data analytics techniques to derive meaningful insights and predictions from complex data.

## Usage
The project is structured as follows:

- **'Database Management.py'**: Script for all database management tasks.
- **'Data Analytics.py'**: Script for data preprocessing, analysis, and modelling tasks.
- **'CarSharing.csv'**: The dataset used in this project.

## Conclusion
This project offers valuable insights into demand patterns for a car-sharing company and showcases the application of data science techniques in real-world scenarios. It provides a foundation for further exploration and analysis in the fields of database management and data analytics.
