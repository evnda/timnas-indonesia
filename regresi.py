import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def prepare_data(df):
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
    return df

def train_and_evaluate(df):
    df = prepare_data(df)
    
    X = df[['word_count', 'hashtag_count']]
    y = df['like_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2, y_test, y_pred

def plot_actual_vs_pred(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='blue', alpha=0.7)
    
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Predict')
    
    ax.set_xlabel('Actual Likes')
    ax.set_ylabel('Predicted Likes')
    ax.set_title('Perbandingan Likes Aktual vs Prediksi')
    ax.legend()
    return fig

def predict_likes(model, word_count, hashtag_count):
    input_data = np.array([[word_count, hashtag_count]])
    return model.predict(input_data)[0]