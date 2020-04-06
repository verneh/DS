import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from collections import Counter
from functions import *
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from numpy import mean
from pprint import pprint
from pywaffle import Waffle

from scipy.stats import uniform, truncnorm, randint

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

# initialize with this.
# data = load_data()
# data.to_csv('data_two.csv')

# Set page outline for sidebar.
pages = ["1. Introduction",
         "2. Data Cleaning",
         "3. EDA",
         "4. Feature Selection",
         "5. Training, Tuning and Model Fitting",
         "6. Final Thoughts",
         ]

page = st.sidebar.selectbox('Navigate', options=pages)

# 1. Introduction
if page == pages[0]:

    st.sidebar.markdown('''

    ---

    - This is a survival analysis / prediction of customer churn based on this kaggle notebook. [[Churn Modelling]](https://www.kaggle.com/shrutimechlearn/churn-modelling)

    - It's a classification dataset containing details of a bank's customers and the target variable is a binary variable reflecting whether the customer left the bank
    (closing his account) or he continued to be a customer.

    ''')

    st.title('Predicting Customer Churn')
    st.write(" ")
    st.image('./references/churn.jpg', use_column_width=True, caption="Image Credit: Sharon McCutcheon on Unsplash")

    '''

    For this streamlit app, we want to find the most suitable model for bank customer that potentially may churn. It's usually evident for subscription-based businesses,
    collections, deployment, and the like. These type of businesses are the ideal applications for machine learning.

    I'm going to write this notebook in a way to learn new things like virtualenvs, pickle, etc. and remind myself how I did it. It might not follow traditional coding conventions.

    The inspiration for this notebook came from [[Nancy Chelaru]] (https://github.com/nchelaru/random-forest-streamlit/blob/master/app.py).

                                                                            - Verne Ornitier
    ###
    '''

#2 Data Cleaning
if page == pages[1]:

    st.sidebar.markdown('''
    ---


    ''')

    section = st.sidebar.radio("Steps:",
                     ('Preview Data',
                      'Remove Customer ID Column',
                      'Re-encode Variables',
                      'Rename Column Headers'))

    st.sidebar.markdown('''

     ---

    So what we're doing here is that we are defining the process for data cleaning step-by-step.

    Feel free to move through the checkboxes to run one code chunk at a time and
    progressively reveal new content!

    ''')

    if section == 'Preview Data':
        st.title('Preview Data')
        st.write(" ")
        '''
        According to the kernel, this dataset belongs to a bank that has hidden its name because of data security. The dataset consists of
        13 attributes and 10,000 rows.

        We will compare a series of models to predict whether a customer will churn based on demographics and purchase history. This data
        can then be used by the marketing and sales departments for their advertising and/or retention campaigns.
        '''

        '''
        ```Python
        import pandas as pd

        df = pd.read_csv("./data/raw/Churn_Modelling.csv")
        ```
        '''

        if st.checkbox("Preview Data?"):

            import pandas as pd

            df = pd.read_csv("./data/raw/Churn_Modelling.csv")

            df.drop(['RowNumber'], axis=1, inplace=True)

            st.dataframe(df.head(5).T)

            st.header(" ")

            '''
            To see a summary of this dataset:

            ```Python

            df.info()

            ```
            '''

            if st.checkbox("TL;DR? Heh."):
                st.dataframe(df_info(df))
                st.markdown('''
                Had to drop some of the rows (Row Number and Surname) and columns (More Numbers, Non-Null Text) in the dataset before it is ready for modeling.
                Row Number doesn't count as an attribute. Will take this step-by-step
                ''')


    if section == 'Remove Customer ID Column':
        st.title("Remove Customer ID Column")
        st.write(" ")
        '''
        The `CustomerID` column contains the unique ID that's associated to a customer. Apparently, to train the model, it is preferable for us to remove this feature.
        '''

        if st.checkbox('Drop Customer ID Column? Got no choice!'):
            df = pd.read_csv("./data/raw/Churn_Modelling.csv")
            df.drop(['CustomerId','Surname', 'RowNumber'], axis=1, inplace=True)
            st.dataframe(df_info(df))
            '''
            `CustomerID` column is removed. 11 attributes remain.
            '''


    if section == 'Re-encode Variables':
        st.title("Re-encode Variables")

        # Something tells me we eventually will need to save a new file for this.
        df = pd.read_csv("./data/raw/Churn_Modelling.csv")

        df.drop(['CustomerId', 'Surname', 'RowNumber'], axis = 1, inplace=True)


        '''
        'Gender` will be encoded by 0s and 1s, presumably corresponding to
        "Females" and "Males".
        '''

        st.dataframe(df.tail(10).style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Gender']]))

        '''
        Let's see the distribution of values in this variable:
        ```Python
        df['Gender'].value_counts()
        ```
        '''

        st.write(df['Gender'].value_counts().to_dict())

        '''

        We will also do the same with the countries, France, Germany, and Spain.
        ```Python
        df['Geography'].value_counts()
        ```
        '''

        st.write(df['Geography'].value_counts().to_dict())

        '''
        Now we try to implement this through our checkbox!
        '''

        with st.echo():
            df['Gender'] = np.where(df['Gender'] == 'Male', 1, 0)


        if st.checkbox("Re-encode? Yes, please."):
            st.dataframe(df.tail(10).style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Gender']]))
            st.markdown('''    ''')
            '''
            For confirmation purposes, we see that the proportion of "Female"/"Male" is the same as "0"/"1" after re-encoding.
            ```Python
            df['Gender'].value_counts()
            ```
            '''

            st.write(df['Gender'].value_counts().to_dict())

            '''
            We will do one hot encoding for the countries. as this seems to be the best way of dealing with it.
            ```Python
            df['Geography'].value_counts()
            ```
            '''

            # custom function for label encoding.
            # geo_label = { ni: n for n,ni in enumerate(set(df['Geography']))}
            # df['Geography'] = df['Geography'].map(geo_label)
            # st.write(df['Geography'].value_counts().to_dict())
            df['Geography'] = pd.Categorical(df['Geography'])
            df_dummies = pd.get_dummies(df['Geography'], prefix='category')
            df = pd.concat([df,df_dummies],axis=1)


            if st.checkbox("Show Code for Encoding?"):
                '''
                ```Python
                data['geography'] = pd.Categorical(data['geography'])
                data_dummies = pd.get_dummies(data['geography'], prefix='c')
                data = pd.concat([data,data_dummies],axis=1)
                '''

    if section == 'Rename Column Headers':
        st.title("Rename Column Headers")
        st.write(" ")
        '''
        A minor point, but for the sake of consistency, we will change 'Geography' to 'Country'.
        '''

        with st.echo():
            df.rename(columns={'Geography': 'Country'}, inplace=True)

        if st.checkbox("Rename Columns?"):
            '''
            Let's do a quick check!
            '''

            st.write(list(df.columns))

            '''
            So Geography is now Country, but why do columns still exist.

            '''




# 3. Data Exploration
if page == pages[2]:

    plt.style.use('seaborn-ticks')

    plt.rcParams.update({'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.title_fontsize': 24, 'legend.loc': 'best', 'legend.fontsize': 'medium',
                         'figure.figsize': (10, 10)})

    # this method loads faster compared to the function for multiple variable dropdown.
    columns_to_skip = ['Unnamed: 0']
    data = pd.read_csv('./data/interim/data_two.csv', usecols=lambda x: x not in columns_to_skip )
    # data = load_data()

    st.sidebar.markdown('''
    ---
    ''')

    # this is the main sidebar for eda.
    section_eda = st.sidebar.radio("Pages within EDA",
                     ("Exploratory Data Analysis", "Sections to Explore", "Findings"))

    if section_eda == 'Exploratory Data Analysis':

        st.sidebar.markdown('''

        ---

        Before diving into model building, it is important to get familiar with the cleaned dataset.

        Let's examine their distributions before examining relationships we want to explore. Tinker around with
        the sidebar and see what you can find out.
        ''')

        st.title('Exploratory Data Analysis')

        '''
        Here we show our plot distributions for each variable.

        '''


        # we will call the function here. will put the function in another file.
        data_numerical = pd.concat([filter_by_dtype(data, int), filter_by_dtype(data, float)], axis=1)

        fig = plt.figure(figsize = (15,15))
        ax = fig.gca()
        data_numerical.hist(ax = ax, bins = 10)
        plt.tight_layout()
        st.pyplot(fig)

        '''
        Age and credit score seems to have a normal distribution. Everything else seems to be uniform in distribution save for the
        categorical fields.
        '''

    if section_eda == 'Sections to Explore':

        st.sidebar.markdown('''

        ---
        ''')

        st.title("Sections to Explore")

        '''
        So what we're going to look at is the relationship of the variables with exited to create an exploratory visualization.
        How about you play around with the variables, through the sidebar?
        '''

        # we will call the function here. will put the function in another file.
        show_histogram_plot(data)

        '''
        0 indicates that the customer did not exit or churn. 1 indicates that the customer left the bank.
        I'm sure you'll be able to find more insights and I'll present mine on the next section.
        '''


    if section_eda == 'Findings':

        st.sidebar.markdown('''

        ---

        ''')

        st.title("Findings")
        # will think about doing graphs here but...will leave it at this.
        '''

        Done playing around? Nice. We also saved our changes here in a new csv as data_two.csv

        To start off, I just want to explore everything that's related to the exited (churn) column and the findings I have are:

        - It is certain that majority of users which is 7,963 (80%) of them - did not churn.

        - In terms of age, median for churners is 36, max at 84, while median for nonchurners is 45, max at 92. Nonchurners seem to be the
        majority until 48 years old.

        - For balance, a way of interpreting this would be "is balance actually referred to as current account balance?" Churners have a higher balance (as
        well as average) while nonchurners have the most at zero. There needs to be some clarification here. If its debt, then you may think that these
        high balance users churned because of financial difficulty. Maybe come up with a program to keep these people (better installment plan and the like)
        or probe further to see if its worth keeping these users. Good job zero debt users!

        - Now, if it's current account balance, then management should come up with a way for people with zero balances to deposit or continue being a customer of the bank.

        - France has 5k user counts. Spain and Germany with close to 2.5k each.

        - Most credit scores were found in the 850 - 854 range. Median for credit score are close at 646 - 654, which is pretty average.

        - 71% have credit cards in which 79% of them are nonchurners. So most likely we have a lot of zero debt users.

        - Estimated salary: There's an average difference of around 3k between churners and nonchurners.

        - As for user activity, 51% are active users.

        - 5,457 are males. 83.5% (4,559) of the males did not churn. There are 4,543 females in which 75% (3,404) of them did not churn.

        - Users have at least 1 - 2 products. These two products consist of 97% of the population.

        - Average tenure is around 5 for both.

        '''



# 4. Feature Selection
if page == pages[3]:

    columns_to_skip = ['Unnamed: 0']
    data_two = pd.read_csv('./data/interim/data_two.csv', usecols=lambda x: x not in columns_to_skip)
    data_two = data_two.drop('country', axis=1)
    # thanks stackoverflow!
    data_two = data_two[[col for col in data_two.columns if col != 'exited']+['exited']]
    data_two.to_csv('./data/processed/train.csv')
    st.sidebar.markdown('''

    ---


    ''')

    st.title('Feature Selection')

    '''
    Since I needed to one hot encode the countries, we had to save our last work and name it as train.csv.

    Here we will look at feature importance. We want to know which of the variables are most relevant
    to our output variable, 'exited'. The higher the score, the more relevant it is.

    We'll be using the KBest class through scikit-learn with its setting set to chi-squared as we have
    no negative features.
    '''
    st.markdown(''' ''')
    X = data_two.iloc[:,0:12]
    y = data_two.iloc[:,-1]

    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    st.write(featureScores.nlargest(10,'Score'))
    st.markdown(''' ''')
    '''
    balance, salary, and age are the top three features. that's good to know. time to normalize the dataset
    for training. For this session, we're going to use all the features and see what happens first.
    '''

# 5. Train, Test and Validation
if page == pages[4]:

    columns_to_skip = ['Unnamed: 0']
    data_train = pd.read_csv('./data/processed/train.csv', usecols=lambda x: x not in columns_to_skip)
    # data_train

    st.sidebar.markdown('''

    ---

    ''')

    # this is the main sidebar for eda.
    section_ttv = st.sidebar.radio("Pages within Training",
                     ("Normalization, Training","Sampling - ROS", "Sampling - SMOTE", "Hyperparameter Tuning", "Modelling"))

    if section_ttv == 'Normalization, Training':

            st.sidebar.markdown('''

            ---

            Brace yourselves. This is going to be one lengthy post.

            ''')
            X = data_train.iloc[:,0:12]  #independent columns
            y = data_train.iloc[:,-1]

            st.title('Normalization, Training')

            st.write(" ")
            '''
            We're going to test all the variables first and see what the results before deciding to remove
            features.

            Before we do that, though we will look at normalize the data.

            We will then only test this with four popular models for churn, logistic regression, decision tree, random
            forest, and xgboost.

            '''

            st.subheader('Normalization')

            st.write(" ")
            '''
            Since we have varying levels in our distribution, we opted to normalize the data. To do that,
            we're just going to use one line of code.

            ```Python
            normalized_X = preprocessing.scale(X)

            '''

            normalized_X = preprocessing.scale(X)

            st.subheader('Training')

            st.write(" ")
            '''
            We can see below that I chose to do a traditional 70-30 split and take a random sample.
            '''
            data_train = data_train.sample(frac=1, replace=False, random_state=1)

            x_train, x_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.30, random_state=42, stratify=data_train['exited'])

            st.write('x_train shape: ', x_train.shape)
            st.write('y_train shape: ', y_train.shape)
            st.write('x_test shape: ', x_test.shape)
            st.write('y_test shape: ', y_test.shape)

            ## Pickle the datasets for later use
            with open('./data/interim/churn.pickle', 'wb') as f:
                pickle.dump([x_train, y_train, x_test, y_test], f)

            data_w = {'Training (negative class)': y_train.value_counts()[0],
                    'Training (positive class)': y_train.value_counts()[1],
                    'Validation':x_test.shape[0]}
            '''
            So after splitting the dataset, we can see that...
            '''
            fig = plt.figure(
                    FigureClass=Waffle,
                    rows=10,
                    columns=10,
                    values=data_w,
                    legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.03), "fontsize":13, 'ncol':3},
                    icons='user',
                    font_size=22,
                    icon_legend=True,
                    figsize=(8, 6)
                )

            st.pyplot(width=800, height=800)
            plt.clf()

            '''
            6,400 are nonchurners. 1,600 are churners. Note: each symbol is equivalent to 100 units. Looks imbalanced.
            Are we oversampling or undersampling?

            '''

    if section_ttv == 'Sampling - ROS':

            st.title('ROS')
            st.write(" ")

            st.subheader('Oversampling with RandomOverSampler')
            '''
            We are oversampling because our CPU can handle the data!
            '''

            pfile = open('./data/interim/churn.pickle','rb')

            x_train, y_train, x_test, y_test = pickle.load(pfile)

            ros = RandomOverSampler(random_state=42)
            x_ros, y_ros = ros.fit_resample(x_train, y_train)

            # summarize class distribution
            st.write(Counter(y_ros))

            ## Pickle the datasets for later use
            with open('./data/processed/random_split_churn.pickle', 'wb') as f:
                pickle.dump([x_ros, y_ros, x_test, y_test], f)

            '''
            ... and now we have 5,594 of x and y. To make sure that it is applied only
            to the training set we will use the imbalanced learn library.
            '''

            seed = 5

            models = []

            models.append(('LR', LogisticRegression()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('RF', RandomForestClassifier()))
            models.append(('XGB', XGBClassifier()))

            # evaluate each model in turn
            results = []
            names = []
            scoring = 'f1_macro'

            for name, model in models:
            	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
            	cv_results = cross_val_score(model, x_ros, y_ros, cv=cv, scoring=scoring)
            	results.append(cv_results)
            	names.append(name)
            	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            	st.write(msg)

            '''
            On this dataset, we use repeated 10-fold cross validation with three repeats to ensure there's no data leakage.
            We're also making sure that it passes through all the data and not just random like KFold. We're using F1 score
            (macro) since this is useful on an uneven distribution. We want to be able to identify positives correctly whether
            a person churns or not.

            '''

            # boxplot algorithm comparison
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            st.pyplot(fig)

            '''
            Ok, so random forest wins. Is this going straight to hyperparameter tuning? What if we do SMOTE?

            '''



    if section_ttv == 'Sampling - SMOTE':

            st.sidebar.markdown('''

            ---

            Here's a link to [[Machine Learning Mastery]](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) which explains which metric to use for your classifier.


            ''')

            st.title('Sampling with SMOTE')
            st.write(" ")

            st.subheader('Oversampling with SMOTE')

            pfile = open('./data/interim/churn.pickle','rb')

            x_train, y_train, x_test, y_test = pickle.load(pfile)

            smote = SMOTE()
            x_smote, y_smote = smote.fit_resample(x_train, y_train)

            counter = Counter(y_smote)
            st.write(counter)
            '''
            So basically, we got the same result as our RandomOverSampler. Let's
            check our model comparison.
            '''
            ## Pickle the datasets for later use
            with open('./data/processed/smote.pickle', 'wb') as f:
                pickle.dump([x_smote, y_smote, x_test, y_test], f)
            # prepare configuration for cross validation test harness
            seed = 7

            # prepare models
            models = []
            models.append(('LR', LogisticRegression()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('RF', RandomForestClassifier()))
            models.append(('XGB', XGBClassifier()))

            # evaluate each model in turn
            results = []
            names = []
            scoring = 'f1_macro'

            for name, model in models:
            	kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
            	cv_results = model_selection.cross_val_score(model, x_smote, y_smote, cv=kfold, scoring=scoring)
            	results.append(cv_results)
            	names.append(name)
            	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            	st.write(msg)


            # boxplot algorithm comparison
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            st.pyplot(fig)

            '''
            XGBoost with the slight edge. We got 84% on our validation set. Guess we're running with RF.
            '''

    if section_ttv == 'Hyperparameter Tuning':

            st.sidebar.markdown('''

            ---

            Even though we have an idea which models to use, we need to fine tune it to get the best scores.
            This will take long to load (like one hour tops!)
            ''')

            st.title('Tuning')
            st.write(" ")

            rfile = open('./data/processed/random_split_churn.pickle','rb')

            x_train, y_train, x_test, y_test = pickle.load(rfile)

            '''
            Decided to define parameters based on scikit learn docs and some [[guidance]](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74).
            and the chosen parameters are:
            '''
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            pprint(random_grid)
            '''
            ```Python
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            pprint(random_grid)

            # random search train
            rf = RandomForestClassifier()
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3,
            verbose=2, random_state=42, n_jobs = -1) # Fit the random search model
            rf_random.fit(x_train, y_train)

            def report(results, n_top=1):

            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    return results['mean_test_score'][candidate], results['std_test_score'][candidate],
                    results['params'][candidate]

            mean_test_score, std_test_score, params_list = report(rf_random.cv_results_, n_top=1)
            ```
            '''

            st.image('./references/mean_std.jpg', use_column_width=False)
            st.image('./references/parameters_2.jpg', use_column_width=False)

            '''
            Decided to run this part in Google Colab due to CPU issues. We opted to do randomsearch because this takes less
            time compared to gridsearch. These are the ideal values identified by randomsearchcv along with mean and std.
            '''
    if section_ttv == 'Modelling':

            st.sidebar.markdown('''

            ---

            Let's see if the data fits.

            ''')

            st.title('Modelling')
            st.subheader('Fit')
            '''
            We will use the parameter list we tuned to evaluate performance of the new rf model, it will be
            used to make predictions on the validation set.

            '''

            '''
            ```Python
             # Define function for calculating scores
            def score(m):
                res = {"Score on training set" : m.score(x_train, y_train),
                       "Score on validation set" : m.score(x_val, y_val)}
                return res

            # Create model object with parameters from random search
            m = RandomForestClassifier(**params_list)
            # Fit model to training set
            m.fit(x_train, y_train)

            # Calculate score
            score(m)
            '''
            st.image('./references/valid.jpg', use_column_width=False)
            '''
            85% on both train and test data.
            Had to separate it using colab since my low spec CPU cannot handle it.
            '''

            st.subheader('Confusion Matrix')
            '''
            This is a confusion matrix to determine churn. To replicate this...

            ```Python
            cm = metrics.confusion_matrix(y_test, y_pred)
            # Build the plot
            plt.figure(figsize=(8,6))
            sns.set(font_scale=1.4)
            sns.heatmap(cm, annot=True, annot_kws={'size':20},
                        cmap=plt.cm.Greens, linewidths=0.2, fmt='d')

            # Add labels to the plot
            class_names_y = ['No Churn (Actual)', 'Churn (Actual)']
            class_names_x = ['No Churn (Predicted)', 'Churn (Predicted)']


            tick_marks = np.arange(len(class_names_y))+0.5

            plt.xticks(tick_marks, class_names_x, rotation=0)
            plt.yticks(tick_marks, class_names_y, rotation=0)
            plt.xlabel('Predicted Label', labelpad=20)
            plt.ylabel('True Label', labelpad=20)
            plt.title('Confusion Matrix for Random Forest Model', pad=20)
            plt.show()
            ```
            '''
            st.image('./references/confusion.jpg', use_column_width=False)

            '''
            So we can see that it has produced 112 false positives (says no churn,
            but actually churned) and 337 false negatives (says churn but actually
            did not churn).
            '''
            st.image('./references/creport.jpg', use_column_width=True)
            '''
            Classification matrix again tells us this is the model that performs the
            best in terms of weighing false positives and negatives equally, which is
            ideal for the F1 score and is what we want. It just so happens to have the
            highest accuracy score as well.
            '''


# 6. Final Thoughts
if page == pages[5]:

            st.sidebar.markdown('''

            ---

            We've come to the end. Thanks for reading. First web notebook app.

            ''')

            st.title('Final Thoughts')
            st.write(" ")

            st.image('./references/predict.jpg', use_column_width=False)
            '''
            Through the predict function, we can determine which of the users will churn
            or not at an 85% accuracy rate. I personally think it doesn't stop here. There's
            a lot more work to be done.

            - For further continuation of this notebook, one thing i thought of is to do
            some feature importance by cutting down the number of features to be used on
            the model. This is needed to see if we can achieve a better score. I'd probably
            experiment with features highlighted in bold as seen below and then redo the
            training, tuning aspects.

            - A separate deep dive on the active users and the countries would be on the to-do
            list.

            '''
            st.image('./references/features.jpg', use_column_width=False)

            '''

            - As for recommendations, I would consider thinking about a program that eases
            payments for older people or those with high account balances. On the same time,
            do some research to induce spending on the younger folks or those with zero balance
            on their accounts. It would be helpful if there was access to campaign data to get
            a better sense of what's happening.

            '''
