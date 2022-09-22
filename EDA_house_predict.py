import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

pwd
cd C:\Users\1248505\Downloads\W207-final-project\house-prices-advanced-regression-techniques
train_file = 'train.csv'
test_file = 'test.csv'

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_data.shape, test_data.shape
# test data has one fewer column than the train data

train_data.columns.difference(test_data.columns)

# SalePrice is the dependent variable that we are predicting

train_data.columns

# visualize the correlation heatmap

plt.figure(figsize = (12,8))
corr = train_data.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# looks like the basement area and 1 st floor space is highly correlated
train_data["TotalBsmtSF"].corr(train_data["1stFlrSF"])

train_data["GarageCars"].corr(train_data["GarageArea"])

# no perfectly multicolinearity

# find out the top 10 correlated variables to 'SalePrice'
plt.figure(figsize = (12,8))
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice', keep = 'first')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(np.tril(cm),
vmin=-1, vmax=1, center=0,
mask=np.triu(cm),
cmap='coolwarm',
annot=True,
square=True,
yticklabels=cols.values,
xticklabels=cols.values)
plt.show()

# point1: garage cars are similar as garage GarageArea
# point 2: total basement SF is similar as 1 floor SF

# point 3: total rooms above gournd highly correlated to living area (introduce avg room SF)
train_data["AvgRmSF"] = train_data["GrLivArea"]/train_data["TotRmsAbvGrd"]
train_data["AvgRmSF"].corr(train_data["SalePrice"])
# higher than total rooms above grade, could be top 10 factors that have strong correlation with saleprice


# take a look at the distribution of these variables

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF']
sns.pairplot(train_data[cols])
plt.show();

# sale price is right skew
# positive linear relationship but the OverallQual might be expotenial

sns.set()
cols = ['SalePrice','GarageCars','FullBath', 'AvgRmSF']
sns.pairplot(train_data[cols])
plt.show();
# car 2,3
# full bath mainly 1 or 2
# positive relationship for avg room SF

# Missing data
missing_count = train_data.isnull().sum().sort_values(ascending = False)
missing_percent = missing_count / (train_data.isnull().count())
missing_percent= missing_percent.sort_values(ascending = False)
missing = pd.concat([missing_count,missing_percent], axis = 1, keys = ['Missing Count', "Missing Percent"])
missing[missing["Missing Count"] >0]

# tell your algorithm not to consider invalid or inexistent values on computations
fill_None = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
"GarageFinish", "GarageCond", "GarageQual", "GarageType",
 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
 "MasVnrType"]
for i in fill_None:
    train_data[i].fillna("None")


sns.displot(train_data["LotFrontage"])

# fillna with the median of LotFrontage in each Neighborhood
train_data["LotFrontage"].fillna(train_data.groupby("Neighborhood")["LotFrotage"].transform("median"))

train_data[train_data["LotFrontage"]==0]
# treat it as None
train_data[train_data["GarageYrBlt"] != train_data["YearBuilt"]][["GarageYrBlt", "YearBuilt", "GarageType"]]
sns.displot(train_data["MasVnrArea"])

# outliers

sns.scatterplot(y = train_data["SalePrice"], x = train_data["GrLivArea"])
pd.set_option('display.max_columns', None)
train_data[train_data["GrLivArea"]>4000].sort_values(by = "SalePrice")
train_data[train_data["SaleCondition"]== 'Partial'].sort_values(by = "SalePrice")
train_data[train_data["Neighborhood"]== 'Edwards'].sort_values(by = "SalePrice")


sns.barplot(x = train_data["SaleCondition"],y =train_data["SalePrice"] )
train_data["SaleCondition"].value_counts()


# preprocessor
all_data = train_data.copy().drop("SalePrice", axis=1)

train_cc.head()
from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X):
    AvgRmSF = X[:, GrLivArea] / X[:, TotRmsAbvGrd]
    TotalSF = X[:, TotalBsmtSF] + X[:, 1stFlrSF] + X[:, 2ndFlrSF]

    return np.c_[X, AvgRmSF, TotalSF]


train_data_cat = train_data.loc[:, train_data.dtypes == object]

# combine train and test data to do preprocessing
train_cc = train_data.copy().drop("SalePrice", axis=1)
test_cc = test_data.copy()

# mark the train set and the test set
train_cc['set'] = 0
test_cc['set'] = 1

# merge two sets together
all_data = pd.concat([train_cc, test_cc], axis=0, copy=True)

all_data.head()


fill_None = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
"GarageFinish", "GarageCond", "GarageQual", "GarageType",
 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
 "MasVnrType", "GarageYrBlt"]

for i in fill_None:
    all_data[i].fillna("None")

all_data["LotFrontage"].fillna(all_data.groupby("Neighborhood")["LotFrontage"].transform("median"))
all_data["MasVnrArea"]=all_data["MasVnrArea"].fillna(0)


all_data.select_dtypes(include=np.number)
all_data.loc[:, all_data.dtypes == object]

all_data_num = all_data.select_dtypes(include=np.number)
all_data_cat = all_data.loc[:, all_data.dtypes == object]

list(all_data_num)
def add_extra_features(X):
    AvgRmSF = X[:, 'GrLivArea'] / X[:, 'TotRmsAbvGrd']
    TotalSF = X[:, 'TotalBsmtSF'] + X[:, '1stFlrSF'] + X[:, '2ndFlrSF']

    return np.c_[X, AvgRmSF, TotalSF]



all_data["AvgRmSF"] = all_data["GrLivArea"]/all_data["TotRmsAbvGrd"]
all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data.head()


train_num = train.select_dtypes(include=np.number)
train_cat = train.loc[:, train.dtypes == object]

num_attribs = list(train_num)
cat_attribs = list(train_cat)

full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

train_prepared = full_pipeline.fit_transform(train)
