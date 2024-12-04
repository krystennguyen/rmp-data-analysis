#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:20:33 2024

@author: krysten
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

SEED = 15113718
TAGS = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]

def main():
    # Load data
    conn = sqlite3.connect('../data/preprocessed/rmp.db')
    num_df = pd.read_sql_query('SELECT * FROM num', conn)
    qual_df = pd.read_sql_query('SELECT * FROM qual', conn)
    tags_df = pd.read_sql_query('SELECT * FROM tags', conn)

    ####################################
    # Q1: Is there a pro-male gender bias in this dataset
    male_ratings = pd.read_sql_query(
        'SELECT avg_rating FROM num WHERE male_clf=1', conn)
    female_ratings = pd.read_sql_query(
        'SELECT avg_rating FROM num WHERE female_clf=1', conn)


    # graph distribution by gender side by side check for normalilty and homoscedasticity
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(male_ratings['avg_rating'], color='lightblue',
                edgecolor='black', bins=100, label='Male')
    plt.title('Average Ratings Distribution for Male Professors', fontsize=12)
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(female_ratings['avg_rating'], color='pink',
                edgecolor='black', bins=100, label='Female')
    plt.title('Average Ratings Distribution for Female Professors', fontsize=12)
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')
    plt.savefig('../results/figures/rating_gender_bias.png')

    # try Welch t-test
    gender_ttest = stats.ttest_ind(
        male_ratings['avg_rating'], female_ratings['avg_rating'], alternative='greater', equal_var=False)
    print("Q1: Welch's T-test for for pro-male bias: ", gender_ttest)
    print('\n')
    ####################################
    # Q2: Is there a gender difference in the spread (variance/dispersion)
    # Levene's test because assumption of normality is violated
    gender_var_levene = stats.levene(
        male_ratings['avg_rating'], female_ratings['avg_rating'], center='median')
    print('Q2: Levene test for gender difference in the spread: ', gender_var_levene)
    print('\n')

    # make our own test statistic with permutation test


    def variance_difference(x, y):
        return np.var(x, ddof=1) - np.var(y, ddof=1)


    gender_var_perm = stats.permutation_test(
        (male_ratings['avg_rating'], female_ratings['avg_rating']),
        statistic=variance_difference,
        alternative='greater',
        permutation_type='independent',
        n_resamples=10000,
        random_state=SEED
    )

    print('Q2: Permutation dress for gender bias in variance difference: ', gender_var_perm)
    ####################################
    # Q3: 95% CI for the above effects  (gender bias in average rating, gender bias in spread of average rating)
    # 1. Gender bias in average rating
    np.random.seed(SEED)
    n_bootstrap = 1000
    np.random.seed(SEED)

    bootstrap_mean_difference = [
        np.mean(np.random.choice(male_ratings['avg_rating'], size=len(male_ratings), replace=True)) -
        np.mean(np.random.choice(
            female_ratings['avg_rating'], size=len(female_ratings), replace=True))
        for _ in range(n_bootstrap)
    ]

    ci_lower, ci_upper = np.percentile(bootstrap_mean_difference, [2.5, 97.5])

    print(
        f'Q3: 95% CI for gender bias in average rating: ({ci_lower}, {ci_upper})')

    # calculate Cohen's d effect size
    mean_diff = np.mean(male_ratings['avg_rating']) - \
        np.mean(female_ratings['avg_rating'])

    # Using sample standard deviation
    std_male = np.std(male_ratings['avg_rating'], ddof=1)
    std_female = np.std(female_ratings['avg_rating'], ddof=1)

    n_male = len(male_ratings)
    n_female = len(female_ratings)
    pooled_std = np.sqrt(((n_male - 1) * std_male**2 +
                        (n_female - 1) * std_female**2) / (n_male + n_female - 2))

    cohen_d = mean_diff / pooled_std
    # sample size
    n = n_male+n_female


    # graph the bootstrap distribution of the means
    plt.figure(figsize=(10, 6))

    sns.histplot(bootstrap_mean_difference, color='lavender',
                edgecolor='black', bins=100)
    plt.title(
        f'95% Confidence Interval of Gender Bias Effect on Average Ratings \n d={round(cohen_d,3)}, N={n}', fontsize=14)
    # red line at ci lower and upper
    plt.axvline(x=ci_lower, color='red', linestyle='--', label='CI Lower')
    plt.axvline(x=ci_upper, color='red', linestyle='--', label='CI Upper')
    plt.xlabel(
        'Mean Difference in Average Ratings between Male and Female Professors)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.savefig('../results/figures/95CI_rating_gender_bias.png')
    print('\n')

    # 2. Gender bias in spread of average rating
    bootstrap_var_difference = [
        np.var(np.random.choice(male_ratings['avg_rating'], size=len(male_ratings), replace=True)) -
        np.var(np.random.choice(female_ratings['avg_rating'], size=len(
            female_ratings), replace=True))
        for _ in range(n_bootstrap)
    ]
    ci_vars_lower, ci_vars_upper = np.percentile(
        bootstrap_var_difference, [2.5, 97.5])
    print(
        f'Q3: 95% CI for gender bias in spread of average rating: ({ci_vars_lower}, {ci_vars_upper}')


    # graph the bootstrap distribution of the variances
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_var_difference, color='lavender',
                edgecolor='black', bins=100)
    plt.title(
        '95% Confidence Interval of Gender Bias in Spread of Average Rating', fontsize=14)
    plt.axvline(x=ci_vars_lower, color='red', linestyle='--', label='CI Lower')
    plt.axvline(x=ci_vars_upper, color='red', linestyle='--', label='CI Upper')
    plt.xlabel(
        'Variance Difference in Average Ratings between Male and Female Professors', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/figures/95CI_rating_gender_bias_spread.png')

    ####################################
    # Q4: Is there a gender difference in the tags awarded by students
    male_tags = pd.read_sql_query(
        'SELECT * FROM tags JOIN num ON tags.id=num.id WHERE num.male_clf=1', conn)
    female_tags = pd.read_sql_query(
        'SELECT * FROM tags JOIN num ON tags.id=num.id WHERE num.female_clf=1', conn)


    # iterate over each tag and perform a chi-square test on the frequency of the tag
    tag_chi2 = {}
    for tag in TAGS:
        tag_chi2[tag] = stats.mannwhitneyu(
            male_tags[tag], female_tags[tag], alternative='greater').pvalue

    # sort the tags by p-value from low to high
    tag_chi2 = dict(sorted(tag_chi2.items(), key=lambda x: x[1]))

    print('Q4: Mann-Whitney U test for gender difference in tags awarded: ')
    print('Top 3 most gendered (lowest p-values):')
    for tag, p in list(tag_chi2.items())[:3]:
        print(f'{tag}: {p}')
    print('Top 3 least gendered (highest p-values):')
    for tag, p in list(tag_chi2.items())[-3:]:
        print(f'{tag}: {p}')
    print('\n')

    ####################################
    # Q5: Is there a gender difference in terms of average difficulty?
    male_diff = pd.read_sql_query(
        'SELECT avg_diff FROM num WHERE male_clf=1', conn)
    female_diff = pd.read_sql_query(
        'SELECT avg_diff FROM num WHERE female_clf = 1', conn)
    conn.close()
    # graph to check for normalilty and homoscedasticity
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(male_diff['avg_diff'], color='lightblue',
                edgecolor='black', bins=100, label='Male')
    plt.title('Average Difficulty Distribution for Male Professors', fontsize=10)
    plt.xlabel('Average Difficulty')
    plt.ylabel('Frequency')
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.histplot(female_diff['avg_diff'], color='pink',
                edgecolor='black', bins=100, label='Female')
    plt.title('Average Difficulty Distribution for Female Professors', fontsize=10)
    plt.xlabel('Average Difficulty')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('../results/figures/difficulty_gender_bias.png')

    # seems normal, Welch t-test
    diff_ttest = stats.ttest_ind(
        male_diff['avg_diff'], female_diff['avg_diff'], alternative='greater', equal_var=False)
    print("Q5: Welch's T-test for gender difference in average difficulty: ", diff_ttest)
    print('\n')
    ####################################
    # Q6: 95% CI for gender difference in average difficulty
    bootstrap_mean_diff = [
        np.mean(np.random.choice(male_diff['avg_diff'], size=len(male_diff), replace=True)) -
        np.mean(np.random.choice(
            female_diff['avg_diff'], size=len(female_diff), replace=True))
        for _ in range(n_bootstrap)
    ]

    ci_lower, ci_upper = np.percentile(bootstrap_mean_diff, [2.5, 97.5])

    print(
        f'Q6: 95% CI for gender difference in average difficulty: ({ci_lower}, {ci_upper})')

    # calculate Cohen's d effect size
    mean_diff = np.mean(male_diff['avg_diff']) - np.mean(female_diff['avg_diff'])

    std_male = np.std(male_diff['avg_diff'], ddof=1)
    std_female = np.std(female_diff['avg_diff'], ddof=1)

    pooled_std = np.sqrt(((n_male - 1) * std_male**2 +
                        (n_female - 1) * std_female**2) / (n_male + n_female - 2))

    cohen_d = mean_diff / pooled_std


    print('\n')
    # graph the bootstrap distribution of the means
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_mean_diff, color='lavender',
                edgecolor='black', bins=100)
    plt.axvline(x=ci_lower, color='red', linestyle='--', label='CI Lower')
    plt.axvline(x=ci_upper, color='red', linestyle='--', label='CI Upper')
    plt.title(
        f'95% Confidence Interval of Gender Bias is Average Difficulty \n d={round(cohen_d,3)}, N={n}', fontsize=14)
    plt.xlabel(
        'Mean Difference in Average Ratings between Male and Female Professors)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig('../results/figures/95CI_difficulty_gender_bias.png')


    ####################################
    # Q7: Build a regression model predicting average rating from all numerical predictors
    # plot to see correlation between predictos
    sns.pairplot(num_df.drop('id', axis=1), size=2)
    plt.title('Pairplot of All Numerical Predictors')
    plt.savefig('../results/figures/numerical_predictors_pairplot.png')

    plt.figure()
    sns.heatmap(num_df.drop('id', axis=1).corr(), annot=True)
    plt.title('Correlation Matrix of All Numerical Predictors')
    plt.savefig('../results/figures/numerical_predictors_heatmap.png')


    # impute prop_take_again with mean
    num_df_imputed = num_df.copy()
    num_df_imputed['prop_take_again'].fillna(
        np.mean(num_df_imputed['prop_take_again']), inplace=True)

    plt.figure()
    sns.heatmap(num_df_imputed.drop('id', axis=1).corr(), annot=True)
    plt.title('Correlation Matrix of Numerical Predictors after Imputation')
    plt.savefig('../results/figures/numerical_predictors_heatmap_imputed.png')


    # drop female_clf because of multicollinearity with male_clf
    X = num_df_imputed.drop(['id', 'avg_rating', 'female_clf'], axis=1)
    y = num_df_imputed['avg_rating']


    # split data to train, validation and test
    np.random.seed(SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED)
    # normalize X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_val_scaled = scaler.transform(X_val)

    # fit linear regression model
    lr_num = LinearRegression()
    lr_num.fit(X_train_scaled, y_train)
    y_train_pred = lr_num .predict(X_train_scaled)
    y_val_pred = lr_num.predict(X_val_scaled)


    print('Q7: Linear Regression (predicting average ratings from numerical predictors):')
    print('-- R2 on Training set: ', r2_score(y_train, y_train_pred))
    print('-- R2 on Validation set: ', r2_score(y_val, y_val_pred))
    print('-- RMSE on Training set:',
        mean_squared_error(y_train, y_train_pred, squared=False))
    print('-- RMSE on Validation set:',
        mean_squared_error(y_val, y_val_pred, squared=False))


    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]

    # Lasso
    lasso_train_rmse = []
    lasso_val_rmse = []

    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train_scaled, y_train)

        # Predict on training and test data for Lasso Regression
        y_train_pred_lasso = lasso_model.predict(X_train_scaled)
        y_val_pred_lasso = lasso_model.predict(X_val_scaled)

        # Calculate MSE for Lasso Regression
        lasso_train_rmse.append(mean_squared_error(
            y_train, y_train_pred_lasso, squared=False))
        lasso_val_rmse.append(mean_squared_error(
            y_val, y_val_pred_lasso, squared=False))

    best_lasso_alpha = alphas[np.argmin(lasso_val_rmse)]

    lasso_model = Lasso(alpha=best_lasso_alpha)
    lasso_model.fit(X_train_scaled, y_train)
    y_train_pred_lasso = lasso_model.predict(X_train_scaled)
    y_val_pred_lasso = lasso_model.predict(X_val_scaled)

    print('Q7: Lasso Regression best alpha: ', best_lasso_alpha)
    print('-- R2 on Training set: ', r2_score(y_train, y_train_pred_lasso))
    print('-- R2 on Validation set: ', r2_score(y_val, y_val_pred_lasso))
    print('-- RMSE on Training set:',
        mean_squared_error(y_train, y_train_pred_lasso, squared=False))
    print('-- RMSE on Validation set:',
        mean_squared_error(y_val, y_val_pred_lasso, squared=False))


    # Ridge
    ridge_train_rmse = []
    ridge_val_rmse = []

    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train_scaled, y_train)

        # Predict on training and test data for Ridge Regression
        y_train_pred_ridge = ridge_model.predict(X_train_scaled)
        y_val_pred_ridge = ridge_model.predict(X_val_scaled)

        # Calculate MSE for Ridge Regression
        ridge_train_rmse.append(mean_squared_error(
            y_train, y_train_pred_ridge, squared=False))
        ridge_val_rmse.append(mean_squared_error(
            y_val, y_val_pred_ridge, squared=False))

    best_ridge_alpha = alphas[np.argmin(ridge_val_rmse)]

    ridge_model = Ridge(alpha=best_ridge_alpha)
    ridge_model.fit(X_train_scaled, y_train)
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)

    print('Q7: Ridge Regression best alpha: ', best_ridge_alpha)
    print('-- R2 on Training set: ', r2_score(y_train, y_train_pred_ridge))
    print('-- R2 on Validation set: ', r2_score(y_val, y_val_pred_ridge))
    print('-- RMSE on Training set:',
        mean_squared_error(y_train, y_train_pred_ridge, squared=False))
    print('-- RMSE on Validation set:',
        mean_squared_error(y_val, y_val_pred_ridge, squared=False))


    # graph betas and corresponding predictors
    plt.figure(figsize=(10, 6))
    plt.plot(lr_num.coef_, marker='o', label='Linear Regression')
    plt.plot(lasso_model.coef_, marker='o', label='Lasso')
    plt.plot(ridge_model.coef_, marker='o', label='Ridge')
    plt.xticks(range(len(X.columns)), X.columns, rotation=45)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Magnitude')
    plt.title('Coefficients for Regression Models Predicting Average Ratings from Numerical Predictors', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/figures/coef_rating_num_predictors.png')


    # test 3 models on test set
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = lr_num.predict(X_test_scaled)
    y_test_pred_lasso = lasso_model.predict(X_test_scaled)
    y_test_pred_ridge = ridge_model.predict(X_test_scaled)

    print('Q7: Test set performance')
    print('-R2:')
    print('-- Linear Regression: ', r2_score(y_test, y_test_pred))
    print('-- Lasso Regression: ', r2_score(y_test, y_test_pred_lasso))
    print('-- Ridge Regression: ', r2_score(y_test, y_test_pred_ridge))
    print('-RMSE:')
    print('-- Linear Regression: ',
        mean_squared_error(y_test, y_test_pred, squared=False))
    print('-- Lasso Regression: ',
        mean_squared_error(y_test, y_test_pred_lasso, squared=False))
    print('-- Ridge Regression: ',
        mean_squared_error(y_test, y_test_pred_ridge, squared=False))

    print('\n')

    # plot ridge regression prediction on strongest predictor (avg_diff)
    avg_diff_predictor = X_test_scaled[:, 0]

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_diff_predictor, y_test, alpha=0.6,
                label='Actual Data', color='blue')
    plt.scatter(avg_diff_predictor, y_test_pred_ridge, alpha=0.6,
                label='Predicted Data', color='orange', marker='x')
    plt.xlabel(f"Strongest Predictor: Average Difficulty (Scaled)")
    plt.ylabel("Average Rating")
    plt.title(
        f"Ridge Regression Predicting Average Rating from Numerical Predictors \n R2: {round(r2_score(y_test, y_test_pred_ridge),3)} - RMSE: {round(mean_squared_error(y_test, y_test_pred_ridge, squared=False),3)}, fontsize=14")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/figures/ridge_avg_rating_num.png')


    ####################################
    # Q8: Build a regression model predicting average ratings from all tags
    # plot to see correlation between tags and average ratings
    sns.heatmap(pd.concat([num_df_imputed['avg_rating'], tags_df.drop(
        'id', axis=1)], axis=1).corr(), annot=False)
    plt.title('Correlation Matrix of Tags and Average Difficulty')
    plt.savefig('../results/figures/tags_ratings_heatmap.png')

    X = tags_df.drop('id', axis=1)
    y = num_df_imputed['avg_rating']

    # split data to train, validation and test
    np.random.seed(SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED)

    # normalize X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # fit Ridge regression model
    ridge_train_rmse = []
    ridge_val_rmse = []

    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train_scaled, y_train)

        y_train_pred_ridge = ridge_model.predict(X_train_scaled)
        y_val_pred_ridge = ridge_model.predict(X_val_scaled)

        ridge_train_rmse.append(mean_squared_error(
            y_train, y_train_pred_ridge, squared=False))
        ridge_val_rmse.append(mean_squared_error(
            y_val, y_val_pred_ridge, squared=False))

    best_ridge_alpha = alphas[np.argmin(ridge_val_rmse)]
    print('Q8: Ridge Regression best alpha: ', best_ridge_alpha)
    ridge_model = Ridge(alpha=best_ridge_alpha)
    ridge_model.fit(X_train_scaled, y_train)
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)

    print('-- R2 on Training set: ', r2_score(y_train, y_train_pred_ridge))
    print('-- R2 on Validation set: ', r2_score(y_val, y_val_pred_ridge))
    print('-- RMSE on Training set:',
        mean_squared_error(y_train, y_train_pred_ridge, squared=False))
    print('-- RMSE on Validation set:',
        mean_squared_error(y_val, y_val_pred_ridge, squared=False))

    # test on test set
    y_test_pred_ridge = ridge_model.predict(X_test_scaled)
    print('Q8: Test set performance')
    print('-R2:', r2_score(y_test, y_test_pred_ridge))
    print('-RMSE:', mean_squared_error(y_test, y_test_pred_ridge, squared=False))

    # graph predictors and coefficients
    plt.figure(figsize=(10, 6))
    plt.plot(ridge_model.coef_, marker='o', label='Ridge')
    plt.xticks(range(len(X.columns)), X.columns, rotation=45, fontsize=8)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Magnitude')
    plt.title('Coefficients for Ridge Regression Predicting Average Ratings from Tags', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/figures/coef_rating_tags.png')

    # strongest predictor is Tough grader
    tough_grader_predictor = X_test_scaled[:, 0]

    plt.figure(figsize=(10, 6))
    plt.scatter(tough_grader_predictor, y_test, alpha=0.6,
                label='Actual Data', color='blue')
    plt.scatter(tough_grader_predictor, y_test_pred_ridge, alpha=0.6,
                label='Predicted Data', color='orange', marker='x')
    plt.xlabel(f"Strongest Predictor: Tough grader (Scaled)")
    plt.ylabel("Average Rating")
    plt.title(
        f"Ridge Regression Predicting Average Rating from Tags \n R2: {round(r2_score(y_test, y_test_pred_ridge),3)} - RMSE: {round(mean_squared_error(y_test, y_test_pred_ridge, squared=False),3)}")

    plt.legend()
    plt.grid(True)
    plt.savefig('../results/figures/ridge_rating_tags.png')
    print('\n')

    ####################################
    # Q9: Build a regression model predicting average difficulty from all tags
    sns.heatmap(pd.concat([num_df_imputed['avg_diff'], tags_df.drop(
        'id', axis=1)], axis=1).corr(), annot=False)
    plt.title('Correlation Matrix of Tags and Average Difficulty')
    plt.savefig('../results/figures/tags_difficulty_heatmap.png')


    X = tags_df.drop('id', axis=1)
    y = num_df_imputed['avg_diff']

    # split data to train, validation and test
    np.random.seed(SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED)

    # normalize X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # fit Ridge regression model
    ridge_train_rmse = []
    ridge_val_rmse = []

    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train_scaled, y_train)

        y_train_pred_ridge = ridge_model.predict(X_train_scaled)
        y_val_pred_ridge = ridge_model.predict(X_val_scaled)

        ridge_train_rmse.append(mean_squared_error(
            y_train, y_train_pred_ridge, squared=False))
        ridge_val_rmse.append(mean_squared_error(
            y_val, y_val_pred_ridge, squared=False))

    best_ridge_alpha = alphas[np.argmin(ridge_val_rmse)]
    print('Q9: Ridge Regression best alpha: ', best_ridge_alpha)
    ridge_model = Ridge(alpha=best_ridge_alpha)
    ridge_model.fit(X_train_scaled, y_train)
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)

    print('-- R2 on Training set: ', r2_score(y_train, y_train_pred_ridge))
    print('-- R2 on Validation set: ', r2_score(y_val, y_val_pred_ridge))
    print('-- RMSE on Training set:',
        mean_squared_error(y_train, y_train_pred_ridge, squared=False))
    print('-- RMSE on Validation set:',
        mean_squared_error(y_val, y_val_pred_ridge, squared=False))

    # test on test set
    y_test_pred_ridge = ridge_model.predict(X_test_scaled)
    print('Q9: Test set performance')
    print('-R2:', r2_score(y_test, y_test_pred_ridge))
    print('-RMSE:', mean_squared_error(y_test, y_test_pred_ridge, squared=False))

    # strongest predictor is Tough grader
    tough_grader_predictor = X_test_scaled[:, 0]

    plt.figure(figsize=(10, 6))
    plt.scatter(tough_grader_predictor, y_test, alpha=0.6,
                label='Actual Data', color='blue')
    plt.scatter(tough_grader_predictor, y_test_pred_ridge, alpha=0.6,
                label='Predicted Data', color='orange', marker='x')
    plt.xlabel(f"Strongest Predictor: Tough grader (Scaled)")
    plt.ylabel("Average Difficulty")
    plt.title(
        f"Ridge Regression Predicting Average Difficulty from Tags \n R2: {round(r2_score(y_test, y_test_pred_ridge),3)} - RMSE: {round(mean_squared_error(y_test, y_test_pred_ridge, squared=False),3)}")

    plt.legend()
    plt.grid(True)
    plt.savefig('../results/figures/ridge_difficulty_tags.png')
    print('\n')


    ####################################
    # Q10: Build a classification model that predicts whether a professor receives a “pepper” from all available factors (both tags and numerical)
    print(num_df['pepper'].value_counts())
    print(num_df['pepper'].value_counts(normalize=True))

    X = pd.concat([num_df_imputed.drop(['id', 'pepper', 'female_clf'],
                axis=1), tags_df.drop('id', axis=1)], axis=1)
    y = num_df['pepper']

    # correlation map for our variables
    sns.heatmap(pd.concat([y, X], axis=1).corr(),
                annot=False, xticklabels=True, yticklabels=True)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Correlation Matrix for Predictors of Pepper')
    plt.savefig('../results/figures/pepper_clf_heatmap.png')


    # split data to train, validation and test
    np.random.seed(SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # normalize X with MinMaxScaler - scale between 0 and 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    # fit a multinomial Logistic Regression
    log_reg = LogisticRegression(multi_class='multinomial')
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    # plot coefficients
    coefficients = log_reg.coef_[0]
    predictors = X.columns
    sorted_indices = np.argsort(coefficients)
    coefficients = np.array(coefficients)[sorted_indices]
    predictors = np.array(predictors)[sorted_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefficients, y=predictors, orient='h')
    plt.title('Logistic Regression Coefficients of Predictors for Pepper', fontsize=14)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Predictors', fontsize=12)
    # Add vertical line at 0 for reference
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('../results/figures/logreg_coef.png')

    plt.figure(figsize=(6, 6))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    plt.savefig('../results/figures/logreg_roc.png')

    optimal_threshold_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_index]
    print('Logistic regression optimal threshold:',optimal_threshold)


    THRESHOLD = optimal_threshold
    y_pred_thres = (y_prob > THRESHOLD).astype(int)

    clf_report = classification_report(y_test, y_pred_thres)
    print(clf_report)

if __name__ == '__main__':
    main()

