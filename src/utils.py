import os

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import seaborn as sns
from ast import literal_eval

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

import glob
import json
import requests
from pprint import pprint

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import RandomizedSearchCV


import multiprocessing as mp

# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from linearmodels.panel import PanelOLS

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from arch.bootstrap import StationaryBootstrap

from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
from scipy.stats.mstats import winsorize

import sklearn.inspection
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import rcParams
# rcParams['figure.figsize'] = 6,6

# import plotly.express as px
import time
import requests
import re
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

def clean_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.1)

def grid_search_cv(param_grid, model, X_train_scaled, y_train, X_test_scaled, y_test):

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters found:", best_params)
    print("Best cross-validation score (negative MSE):", grid_search.best_score_)

    model_pred = best_model.predict(X_test_scaled)
    model_rmse = np.sqrt(mean_squared_error(y_test, model_pred))
    model_r2 = r2_score(y_test, model_pred)

    print(f"Test RMSE: {model_rmse:.4f}")
    print(f"Test R²: {model_r2:.4f}")
    
    return best_params, best_model

def prepare_data(df, features, target_col='next_day_3_log_return'):

    data = df.copy()
    
    data = data.dropna(subset=[target_col])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    X = data[features]
    y = data[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['int64', 'float64']:
#                 X[col].fillna(X[col].median(), inplace=True)
#                 X[col].fillna(X[col].mean(), inplace=True)
                X[col].fillna(0., inplace=True)
    
    return X, y

def data_train_test_split(df, features, target_col, q=2, test_size=0.1):
    X, y = prepare_data(df, features, target_col=target_col)

    y_bins = pd.qcut(y, q=q, labels=False, duplicates='drop')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, 
#         stratify=y_bins
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(round(y_train.mean(), 3), round(y_test.mean(), 3))
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def scatter_regplot(x, y, **kws):
    ax = plt.gca()
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, edgecolor='w', linewidth=0.5)
    # Линия регрессии
    sns.regplot(x=x, y=y, scatter=False, 
                line_kws={'color': 'red', 'alpha': 0.8, 'linewidth': 2})
    r, p_value = stats.pearsonr(x, y)
    ax.annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=9)

def diag_kde(x, **kws):
    ax = plt.gca()
    sns.histplot(x, kde=True, stat="density", alpha=0.7, ax=ax)

    mean_val = x.mean()
    median_val = x.median()
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
    if ax.get_subplotspec().is_first_row() and ax.get_subplotspec().is_first_col():
        ax.legend(loc='upper right', fontsize=8)

def lower_hexbin(x, y, **kws):
    ax = plt.gca()
    hb = ax.hexbin(x, y, gridsize=30, cmap='Blues', alpha=0.8, mincnt=1)
    plt.colorbar(hb, ax=ax, shrink=0.7)
    
    
def calculate_pvalues(df, method='pearson'):
    n = df.shape[1]
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                p_matrix[i, j] = 0
            else:
                if method == 'pearson':
                    corr, p_value = pearsonr(df.iloc[:, i], df.iloc[:, j])
                    p_matrix[i, j] = p_value
                elif method == 'spearman':
                    corr, p_value = spearmanr(df.iloc[:, i], df.iloc[:, j])
                    p_matrix[i, j] = p_value
    return pd.DataFrame(p_matrix, index=df.columns, columns=df.columns)

def annotate_with_pvalues(corr_matrix, p_matrix):
    annot_matrix = np.empty_like(corr_matrix, dtype=object)
    n = corr_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                annot_matrix[i, j] = '1.00'
            elif mask[i, j]: 
                annot_matrix[i, j] = ''
            else:
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_matrix.iloc[i, j]
                
                stars = ''
                if p_val < 0.01:
                    stars = '***'
                elif p_val < 0.05:
                    stars = '**'
                elif p_val < 0.1:
                    stars = '*'
                
                annot_matrix[i, j] = f'{corr_val:.2f}{stars}'
    
    return annot_matrix

def plot_corr_matrix(corr_matrix, mask, annotations, title='Correlation with p-values'):
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(corr_matrix,
                mask=mask,
                annot=annotations,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='', 
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 9})

    plt.title(title, fontsize=16, pad=20)

    legend_text = '\n'.join([
        '*** p < 0.01',
        '** p < 0.05', 
        '* p < 0.1',
    #     'Без звездочек: p ≥ 0.05'
    ])
    plt.figtext(0.72, 0.92, legend_text, fontsize=10, 
               bbox=dict(boxstyle="round, pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

def get_coordinates_photon(zip_code, state=None):
    try:
        if state:
            query = f"{zip_code}, {state}, USA"
        else:
            query = f"{zip_code}, USA"
        
        url = f"https://photon.komoot.io/api/?q={query}&limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data['features']:
            coords = data['features'][0]['geometry']['coordinates']
            return coords[1], coords[0]  # lat, lon
        return None, None
    except Exception as e:
        print(f"Error geocoding {zip_code}: {e}")
        return None, None

def get_coordinates_with_retry(zip_code, state, max_retries=3):
    for attempt in range(max_retries):
        try:
            lat, lon = get_coordinates_photon(zip_code, state)
            if lat and lon:
                return lat, lon
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {zip_code}: {e}")
            time.sleep(2)
    return None, None

def parse_industry_file_with_expanded_codes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    industry_blocks = re.split(r'\n\s*\n', content.strip())
    
    data = []
    
    for block in industry_blocks:
        lines = block.strip().split('\n')
        
        if len(lines) < 2:
            continue
        
        first_line = lines[0].strip()
        match = re.match(r'(\d+)\s+(\w+)\s+(.+)', first_line)
        
        if match:
            industry_num = int(match.group(1))
            industry_code = match.group(2)
            industry_name = match.group(3).strip()
            
            # Обрабатываем диапазоны кодов
            for line in lines[1:]:
                line = line.strip()
                if line and re.match(r'\d{4}-\d{4}', line):
                    start, end = map(int, line.split('-'))
                    
                    # Создаем отдельную запись для каждого кода в диапазоне
                    for code in range(start, end + 1):
                        data.append({
                            'industry_number': industry_num,
                            'industry_code': industry_code,
                            'industry_name': industry_name,
                            'sic_code': code
                        })
    
    return pd.DataFrame(data)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix,
#             mask=mask,
#             annot=True,
#             cmap='RdBu_r',
#             center=0,
#             square=True,
#             fmt='.2f',
#             linewidths=0.5,
#             cbar_kws={'shrink': 0.8})

# plt.title('Correlation', fontsize=16, pad=20)
# plt.tight_layout()
# plt.show()