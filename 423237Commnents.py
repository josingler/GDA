#!/usr/bin/env python
# coding: utf-8
# <a href="https://colab.research.google.com/github/josingler/GDA/blob/main/423237.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# We have provided a comma-separated file with this package which includes about **3700** instances with **18** **integer** and **real** attributes. 
# 
# The data in the file have 1 record per line. The target feature (**“Revenue”**) is the last feature (column) in each line. Your task is to do some preprocessing steps as described below and to measure the accuracy of different machine learning methods on predicting the target variable according to the steps that are described below. Note that there is only a single target variable for each record. 
# 
# The dataset consists of **10** numerical, **7** categorical attributes and one target feature (classification label). 
# 
# **Numerical** features include: 
# 
# * Administrative
# * Administrative_Duration
# * Informational
# * Informational_Duration
# * ProductRelated
# * ProductRelated_Duration
# * BounceRates
# * ExitRates
# * PageValues
# * SpecialDay
# 
# Also, the **categorical** features include:
# 
# * Month
# * OperatingSystems
# * Browser
# * Region
# * TrafficType
# * VisitorType
# * Weekend
# 
# Each of the tasks below will be evaluated independently of the others, so you must send us your results (code and output, if applicable) from each step at the end. The processing is done with Python 3. Please send us the code and the respective output in one file (the downloaded Jupyter Notebook). If you do not send us the code for a step and, where required, an output, the step will be considered as not processed.
# 
# **In some steps you are asked to output the head of a dataframe. Then you have to output the complete header (column names) of the data frame and the first 3 rows (data records).**
# 
# 
# 
# 
# # Please run the followig piece of code to load the dataset into the **"dataset"** variable. You can use and manipulate data using this object
# # Then, Carry out the following steps with the provided data and document the number of each step in a python comment:
# In[ ]:
# # 1.   Print out the head name of features.
# 
# In[ ]:
# # 2.   Print out the first and last rows of the dataset.
# In[ ]:
# # 3.   Check if there is any missing value in the data. Here it is sufficient to specify the code, output is not necessary.
# In[ ]:
# 
# In[ ]:
# # 4.   Exchange the positions of the **10th** and **11th** column with each other. Then output the header of the dataframes.
# In[ ]:
#print(colu_List)
# # 5.   Change the name of "**VisitorType**" feature to "**new_VisitorType**".
# 
# In[ ]:
# # 6.   Print out the row index of the cell with the highest **“ProductRelated_Duration”** value.
# In[ ]:
#dataset[dataset['ProductRelated_Duration'].max()].index
#dataset[dataset['ProductRelated_Duration']==dataset['ProductRelated_Duration'].max()]
# # 7.   Please remove the **“Informational”** feature from the dataset. Then output the header of the dataframe.
# In[ ]:
# # 8.   Print out the first and third quantile of **“PageValues”** feature.
# In[ ]:
# # 9.   Output the number of missing values in each column.
# In[ ]:
# # 10.   From the dataset, filter every 250th row in the data for the features **“Month”**, **“ExitRates”** and **“SpecialDay”**, starting from row 0.
# In[ ]:
# # 11.   Scales and transfer the values of all of the numerical features in the range of 0 to 5, using min-max normalization method. Then output the first 10 lines of the dataframe. 
# In[ ]:
# # 12.   Print out the head name and variance of the numerical feature with highest variance
# In[ ]:
# # 13.   Use the “apply” function (or an equivalent function) to replace missing values in the columns "**Informational_Duration**" and "**ProductRelated_Duration**" with the **mean value** of the respective  attribute.
# In[ ]:
# # 14.   Print out the index of rows where values of **“OperatingSystems”**, **“Browser”** and **“TrafficType”** columns match.
# In[ ]:
# # 15.   Calculate and output the value of the correlation between the   "**Informational_Duration**" and "**ProductRelated_Duration**" features.
# In[ ]:
# # 16.   In the **“Browser”** feature, keep only top 4 most frequent values as it is and replace the other values with the ***“Other”*** label.
# In[ ]:
# # 17.   Compute and print out the mean of **“BounceRates”** of each **“Month”**.
# In[ ]:
# # 18.   Print out all of the instances with **“BounceRates”** values greater than **“ProductRelated_Duration”**.
# In[ ]:
# # 19.   Please plot the boxplot graph of all numerical features.
# In[ ]:
# # 20.   Since the provided data is imbalanced, use upsampling approach to make it balanced.
# In[ ]:
# # 21.   Use 3 different classification approaches to estimate the target characteristic ("**revenue**") from the pre-processed data. Use a 5-fold cross-validation for each of these approaches. The output is described in the next step.
# In[ ]:
# #22.  Output the confusion matrix of the results from **step 21** and the values for the F1 measure of the trained classifiers.
# In[ ]:
