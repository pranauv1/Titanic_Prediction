import pandas as pd
import numpy as np
import missingno
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

#Downloading Datasets
! kaggle competitions download -c titanic

#Loading in data
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
sample_sub = pd.read_csv("/content/gender_submission.csv")

#Reading those files to get an idea of structure
train.head()

test.head()

#This is how the submission should look like
sample_sub.head()

print('train shape:', train.shape)
print('test shape:', test.shape)
print('sample submission shape:', sample_sub.shape)


#Checking non_null and data types of dataset train
train.info()

#Checking non_null and data types of dataset test
test.info()

#Checking missing data in trainig set by columns
train.isnull().sum().sort_values(ascending = False)

missingno.matrix(train)

#Checking missing data in test set by columns
test.isnull().sum().sort_values(ascending = False)

missingno.matrix(test)


#Will go through some text variables

#Survival chance by 'sex'
train[['Sex', 'Survived']].groupby('Sex', as_index = False).mean().sort_values(by = 'Survived', ascending = False)

#Female people are more likely to survive


#Survival chance by 'Pclass' (Passenger Class)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

#1st class people are more likely to survive or they were prioritized while evacuating


#Survival chance by 'Pclass' (Passenger Class) and 'Sex'

graph = sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = train, kind = 'bar')
graph.despine(left = True)
plt.ylabel('Survival chance')
plt.title('Survival chance by Sex and Passenger Class')


#Survival chance by 'Embarked'
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

#People embarked at location "C" are more likely to survive


#Will get a chart by passenger class and emabarked location
sns.factorplot('Pclass', col = 'Embarked', data = train, kind = 'count')

#Most of the people are embarked at location 'S' and are 3rd class


#Now will go through some numeric variables

#Detect and remove outliers in numerical variables
def detect_outliers(df, n, features):
    """"
    This function will loop through a list of features and detect outliers in each one of those features. In each
    loop, a data point is deemed an outlier if it is less than the first quartile minus the outlier step or exceeds
    third quartile plus the outlier step. The outlier step is defined as 1.5 times the interquartile range. Once the 
    outliers have been determined for one feature, their indices will be stored in a list before proceeding to the next
    feature and the process repeats until the very last feature is completed. Finally, using the list with outlier 
    indices, we will count the frequencies of the index numbers and return them if their frequency exceeds n times.    
    """
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR 
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(key for key, value in outlier_indices.items() if value > n) 
    return multiple_outliers

outliers_to_drop = detect_outliers(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
print("drop these {} indices: ".format(len(outliers_to_drop)), outliers_to_drop)


#Those indices in numeric values
train.loc[outliers_to_drop, :]

#Drop those indices from the train set
train = train.drop(outliers_to_drop, axis = 0).reset_index(drop = True)

#Now will get a heat map with numeric variables
sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')
#And it seems that 'Fare' is the only variable that affects chances of survivel



#Survival chance by 'SibSp' (Sibling and Spouse)
train[['SibSp', 'Survived']].groupby('SibSp', as_index = False).mean().sort_values(by = 'Survived', ascending = False)
#People with 1 sibling/spouse are more likely to survive



#Survival chance by 'Parch' (Parent and Child)
train[['Parch', 'Survived']].groupby('Parch', as_index = False).mean().sort_values(by = 'Survived', ascending = False)
#People with 3 children/Parents are more likely to survive



#Null values in age
train['Age'].isnull().sum()

#Survival chance by age
sns.kdeplot(train['Age'][train['Survived'] == 0], label = 'Did not survive')
sns.kdeplot(train['Age'][train['Survived'] == 1], label = 'Survived')
plt.xlabel('Age')
plt.title('Passenger Age Distribution by Survival')


#Null values in 'Fare'
train['Fare'].isnull().sum()




###################### Data Preprocessing ##############################

#Drop the 'Ticket' and 'Cabin' from datasets
  #dropping the ticket column since we have the 'Fare' column
  #dropping the cabin column since it containes a lot of null values
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)


#Now will see the missing(null) values in the training set
train.isnull().sum().sort_values(ascending = False)


#Okay will fill in 'Embarked' with the most frequent value in the set
freq_e = train['Embarked'].dropna().mode()[0]
train['Embarked'].fillna(freq_e, inplace = True)


#Now will see the missing(null) values in the test set
test.isnull().sum().sort_values(ascending = False)


#Okay will fill in 'Fare' with the median value of 'Fare' in the set
median = test['Fare'].dropna().median()
test['Fare'].fillna(median, inplace = True)


#Let's combine both train and test sets
combination = pd.concat([train, test], axis = 0).reset_index(drop = True)
combination.head()


#Missing values in combined dataset
combination.isnull().sum().sort_values(ascending = False)
#We can ignore the 'Survived' missing values since it's from the test set


#Converting the 'Sex' column into numerical values
combination['Sex'] = combination['Sex'].map({'male': 0, 'female': 1})



#Time to fill in the age column
age_null_counts = list(combination[combination['Age'].isnull()].index)
len(age_null_counts)


'''
This loop wil go through each age in the list(age_null_counts) and it will locate the rows that has the same 
'SibSp', 'Parch' and 'PClass' values after it will fill the missing ages with the median of those rows, if no
rows found it will fill the age with the median of 'Age' column
'''

for i in age_null_counts:
  median_age = combination['Age'].median()
  predicted_age = combination['Age'][(combination['SibSp'] == combination.iloc[i]['SibSp'])&
                                     (combination['Parch'] == combination.iloc[i]['Parch'])&
                                     (combination['Pclass'] == combination.iloc[i]['Pclass'])].median()
  if np.isnan(predicted_age):
    combination['Age'].iloc[i] = median_age
  else:
    combination['Age'].iloc[i] = predicted_age
    
#Check if it worked
combination['Age'].isnull().sum()
combination.head()



#Time to tranform those data into numeric values
#Plotting 'Fare' distribution
sns.distplot(combination['Fare'], label = 'Skewness: %.2f'%(combination['Fare'].skew()))
plt.legend(loc = 'best')
plt.title('Fare Distribiution')


#will reduce skewness
combination['Fare'] = combination['Fare'].map(lambda x : np.log(x) if x > 0 else 0)


#Plotting 'Fare' distribution after
sns.distplot(combination['Fare'], label = 'Skewness: %.2f'%(combination['Fare'].skew()))
plt.legend(loc = 'best')
plt.title('Fare Distribiution After Log')

combination.head()


#Will get 'Title' separated from 'Name'
combination['Title'] = [name.split(',')[1].split('.')[0].strip() for name in combination['Name']]
combination[['Name', 'Title']].head()

combination['Title'].value_counts()

#Sorting those titles
combination['Title'] = combination['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Jonkheer', 'Don', 'Capt',
                                                     'the Countess', 'Sir', 'Dona'],'Rare')
combination['Title'] = combination['Title'].replace(['Mlle', 'Ms'], 'Miss')
combination['Title'] = combination['Title'].replace('Mme', 'Mrs')

sns.countplot(combination['Title'])


#Survival chance by 'Title'
combination[['Title', 'Survived']].groupby(['Title'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

#Now we don't need 'Name' column anymore, will drop it
combination = combination.drop('Name', axis=1)
combination.head()


#Will calculate family size of passenger using 'SibSp', 'Parch'
combination['FamilySize'] = combination['SibSp'] + combination['Parch'] + 1
combination[['SibSp', 'Parch', 'FamilySize']].head()


#Chances of survival by 'FamilySize'
combination[['FamilySize', 'Survived']].groupby('FamilySize', as_index = False).mean().sort_values(by = 'Survived', ascending = False)


#Will create a new column to define if the passenger is alone or not
combination['IsAlone'] = 0
combination.loc[combination['FamilySize'] == 1, 'IsAlone'] = 1
combination.head()


#Chances of survival by 'IsAlone'
combination[['IsAlone', 'Survived']].groupby('IsAlone', as_index = False).mean().sort_values(by = 'Survived', ascending = False)


#Time to drop 'FamilSize', 'SibSp' and 'Parch' from dataframe
combination = combination.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)
combination.head()


#Will create a new column 'AgeBand' --> get the age and group them into 5 groups
combination['AgeBand'] = pd.cut(combination['Age'], 5)
combination.head()

#Chances of survival by 'AgeBand'
combination[['AgeBand', 'Survived']].groupby('AgeBand', as_index=False).mean().sort_values(by = 'AgeBand')

#will convert the 'Age' column to groups from 0-4 using the values from 'AgeBand'
combination.loc[combination['Age'] <= 16.136, 'Age'] = 0
combination.loc[(combination['Age'] > 16.136) & (combination['Age'] <= 32.102), 'Age'] = 1
combination.loc[(combination['Age'] > 32.102) & (combination['Age'] <= 48.068), 'Age'] = 2
combination.loc[(combination['Age'] > 48.068) & (combination['Age'] <= 64.034), 'Age'] = 3
combination.loc[(combination['Age'] > 64.034), 'Age'] = 4

combination.head()


#Will drop 'AgeBand'
combination = combination.drop('AgeBand', axis = 1)


#Will convert data type of 'Age' into an integer
combination['Age'] = combination['Age'].astype('int32')
combination['Age'].dtype

#Will create a new column by multiplying 'Age' and 'Pclass'
combination['Age*Class'] = combination['Age'] * combination['Pclass']
combination.head()



#Converting 'Title', 'Embarked' and 'Fare' into ordinal values
#First will encode 'Title' and 'Embarked'
combination = pd.get_dummies(combination, columns = ['Title'])
combination = pd.get_dummies(combination, columns = ['Embarked'], prefix = 'Em')
combination.head()

#Now will divide 'Fare' into 4 groups like what we did to the age
combination['FareBand'] = pd.cut(combination['Fare'], 4)

#Survivel chances of 'FareBand'
combination[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by = 'FareBand')

#will convert the 'Fare' column to groups from 0-3 using the values from 'FareBand'
combination.loc[combination['Fare'] <= 1.56, 'Fare'] = 0
combination.loc[(combination['Fare'] > 1.56) & (combination['Fare'] <= 3.119), 'Fare'] = 1
combination.loc[(combination['Fare'] > 3.119) & (combination['Fare'] <= 4.679), 'Fare'] = 2
combination.loc[combination['Fare'] > 4.679, 'Fare'] = 3

combination.head()


#Will drop 'FareBand' column
combination = combination.drop('FareBand', axis = 1)

#Converting 'Fare' into integer
combination['Fare'] = combination['Fare'].astype('int32')
combination.head()




#Time to separate train and test from combination
train = combination[:len(train)]
test = combination[len(train):]

train.head()


#will drop 'PassngerId' from training set
train = train.drop('PassengerId', axis = 1)
train.head()


#Converting survived to integer in train set
train['Survived'] = train['Survived'].astype('int32')
train.head()

#Time for test set
test.head()


#Drop the 'Survived' from test set
test = test.drop('Survived', axis = 1)
test.head()



#Ah Finally! time to model
#Will split train data
X_train = train.drop('Survived', axis = 1)
Y_train = train['Survived']

X_test = test.drop('PassengerId', axis = 1).copy()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)



#Will go with Random Forest
rd_tree = RandomForestClassifier(n_estimators=100)
rd_tree.fit(X_train, Y_train)
Y_pred = rd_tree.predict(X_test)
acc_de_tree = round(rd_tree.score(X_train, Y_train) * 100, 2)
print(acc_de_tree)


#Tuning
cv_results = cross_val_score(RandomForestClassifier(), X_train, Y_train, scoring = 'accuracy', cv = 10)

print(cv_results.mean())


#Preparing for Kaggle submission
print(len(Y_pred))


sample_sub.head()

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_pred})
submission.head()


submission.shape

#Save the CSV file
submission.to_csv('/content/submission.csv', index = False)
