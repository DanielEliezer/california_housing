# california_housing

Check out the medium post about this project:* https://danieleliezer.medium.com/end-to-end-maching-learning-project-predicting-house-prices-in-california-2e95171d49cc

**- Description and Motivation** <br/>
Welcome to the California Housing Prices Analysis! In this project, we are going to use the 1990 California Census dataset to study and try to understand how the different attributes can make the house prices get higher or lower. How does the location impact? How about the size of the house? The age?

This dataset has a lot of information that can help us. The main goal is to build a Machine Learning Model in python that can learn from this data and make predictions of the median price of a house in any district, given all the other metrics provided in the dataset.

**- Main Results**

• Creating a new 'cluster' feature using KMeans helped the model a lot. <br/>
• The model with the best performance was the XGBoosting. After the hyperparameters optimization, this model outperformed the Random Forest by a lot (R² from 0.825 to 0.844) <br/>
• The R² of the final model is 0.844. The RMSE is $38059 <br/>
• The location of the house turned out to be the most important kind of feature to the final model.

**- Personal Challenges**

• Use the mapbox API to plot a map <br/>
• Use the datapane API to plot the result in the medium as an interactive map (this step is not in the code) <br/>
• Compare the result of different kind of models using log and standard scaler <br/>
• Use the KMeans in the feature engineer process. <br/>
• Create a custom class to feed it to the Pipeline as a Transformation Step <br/>
• Use the baesyan optimization to tune the hyperparameters. 


**- Files in the repository:**

• housing.csv: A dataset with 20640 rows, and each one of them stores information about a specific block, such as median house price, median income of the familys, size of the house, location, etc. <br/>
• CaliforniaHousing.ipynb: The notebook of the project <br/>
• CaliforniaHousing.py. The project as a python file.

**- Libraries used:**
Pandas, Numpy, Seaborn, Matplotlib, Plotly, Autoviz, Scipy, Sklearn, XGBoost, Skopt

**- External tools:**
Mapbox Api, Datapane Api

*PS: This project is part of the Udacity Nanodegree Course*
