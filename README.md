# House-Price-Prediction

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Kaggle [link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

![image](https://user-images.githubusercontent.com/59941969/191819673-5518a036-176e-44a5-9777-fa0e8ba1989d.png)

Data set size: 1460 observations, 80 features ( 23 nominal, 23 ordinal, 34 numeric) 

![image](https://user-images.githubusercontent.com/59941969/191820422-f68b7e3c-107d-4666-b4a1-0866b38083fa.png)

![image](https://user-images.githubusercontent.com/59941969/191819749-1df71900-9689-4581-8c61-c5ddf4fe6db2.png)


In the baseline models, Lasso regression gives us the lowest error. Therefore, we did our error analysis based on the lasso model. For the complexity vs performance model, the upper left graph on the screen, we plotted the number of features on the x-axis and the Root mean squared errors on the y-axis. As we can tell from the graph, when the number of features is low, the cross validation performance is very similar as training performance. When the number of features is above 30, we are seeing training set over performand the cross validation set, which shows that the model starts to overfit the dataset. 

After knowing the fitting situation of the model, we took a look at the top 10 rows which have the biggest error, and plotted the error against top indicators that we found out in EDA. On the bottom, the error is distributed normally, but there are some points on the left side of the histogram, which has relatively high errors. For these two points, which you can see in the box, they have high overall quality, and large living area, but have relatively low price. These homes were sold in 2007 and 2008 and their sale condition is Partial. We decided to remove these two outliers. We plotted the median sale price for each sold year, which is on the upper right corner. We found that it is not a linear relationship between year sold and sale price. Year sold was treated as a numeric variable in the dataset, and now we transferred it to categorical variable. 

![image](https://user-images.githubusercontent.com/59941969/191819853-cae7dbf5-c95b-40a3-9731-ebfc1b7aaa84.png)

In order to have a better prediction, we added four new features to aggregate information, including total square foot, and average room sqft and total bathrooms, total porch square foot. 

Other than adding few features, we created binary features to simplify some skewed numeric features. 90% of houses in the dataset do not have a pool, and 5% of houses do not have a garage. We plotted the histogram and found that the zero has relatively high counts. Therefore, we transformed them into binary variables. 

We also use the year sold minus year remodelled to indicate how new the equipment is in the house. The more recent the house has been remodelled, the higher the price tends to be. 

We also turned the month sold to season feature, such as winter, spring, etc in order to capture the seasonality.

We also measured the features which have high skewness and use the log function to make them distributed normally. 

Lastly, we added neighborhood bins which groups neighborhoods based on the median sale price. 


![image](https://user-images.githubusercontent.com/59941969/191819914-cabb94e4-4aaf-4336-936e-bf582506965a.png)

Based on error analysis, we removed two outliers and added new features. In this section, we built models based on the improved dataset.

The first model is the Lasso model in baseline analysis. It is a great choice for modeling the problem because SalePrice has inherent linear relationship with many features, as we observed in EDA. Furthermore, given the large number of features (260) and small number of observations (1460), Lasso performs feature selection, which improves data density and coefficient estimation. With the improved data set, Lasso selected 93 features, and our RMSE improved from 0.137 to 0.121.

Our second model is based on the idea that expensive homes might have different feature weights from cheap homes. We calculated median SalePrice by neighborhood and split the neighborhoods into three bins, cheap homes, average homes, and expensive homes. Then, we fit one Lasso model for each neighborhood bin. However, the RMSE is not as good as a single Lasso model. We think it is because of data sparsity as we have divided one small dataset into three.

Next, we fitted a random forest model. The performance is worse than the linear models. We think it’s because the linear assumption is a very useful insight, which random forest model is not using. However, this model still proves useful when we ensemble with Lasso, because it looks at the data in a different way.

Finally, we make an ensemble model. To learn the mixing weights, we use the dev labels, three sets of model predictions and linear regression. We find that 3 part Lasso and 1 part random forest make a nice cocktail. The split lasso model only gets allocated minimal weight, so we do not use it in the final ensemble. The final ensemble model gives us RMSE of 0.119.

![image](https://user-images.githubusercontent.com/59941969/191819952-73326873-c4d8-479d-886c-efd9e737e001.png)


Now we prepare everything for submission. We combine train and dev data sets, fit lasso and random forest models, mix them 3 to 1, and make predictions on the test data set. Kaggle gives us RMSE of 0.1217, which is slightly higher than our dev performance. This puts us at 10th percentile on the leadership board.


![image](https://user-images.githubusercontent.com/59941969/191819982-ccbfd4d3-c900-4e4e-bcdc-9ecfb7198a8c.png)
