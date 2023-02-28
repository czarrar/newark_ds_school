import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

@st.cache_data
def load_data():
    with st.spinner("Loading data..."):
        dat = pd.read_csv('movies_train.csv')

        # Replace NAs with median
        dat['runtime'] = dat['runtime'].fillna(dat['runtime'].median())

        # Get the columns
        numeric_columns = ['budget','popularity','runtime','revenue']
        feature_columns = [c for c in numeric_columns if c!='revenue']

        X_train, X_test, y_train, y_test = train_test_split(
            dat[feature_columns], dat[['revenue']], test_size=0.2, random_state=42)

    return dat, X_train, X_test, y_train, y_test


def plot_reg():

    st.markdown("# Model Complexity")
    st.sidebar.header("Model Complexity")
    st.write(
        """TODO"""
    )
    st.markdown("Note: r^2 tells us how much variation our line explains of the data. It can be 0 to 100%.")

    # Get the data
    dat, X_train, X_test, y_train, y_test = load_data()

    X_train = X_train.sort_values('budget').reset_index(drop=True)
    X_test = X_test.sort_values('budget').reset_index(drop=True)

    budget_mat_train = X_train[['budget']]/1e8
    budget_train = X_train['budget']/1e8
    revenue_train = y_train['revenue']/1e8

    budget_mat_test = X_test[['budget']]/1e8
    budget_test = X_test['budget']/1e8
    revenue_test = y_test['revenue']/1e8

    poly = PolynomialFeatures(degree=int(st.session_state['complexity']))
    X_ = poly.fit_transform(budget_mat_train)
    predict_ = poly.fit_transform(budget_mat_test)
    clf = Ridge(alpha=1e-3)
    #clf = LinearRegression()
    clf.fit(X_, revenue_train)
    y_train_pred = clf.predict(X_)
    y_test_pred = clf.predict(predict_)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Horizontally stacked subplots')

    ax1.title.set_text('Training Data')
    ax1.scatter(budget_train, revenue_train)
    ax1.plot(budget_train, y_train_pred, color='k')

    ax2.title.set_text('Test Data')
    ax2.scatter(budget_test, revenue_test)
    ax2.plot(budget_test, y_test_pred, color='k')

    train_r2 = r2_score(revenue_train, y_train_pred).round(4) * 100
    test_r2  = r2_score(revenue_test, y_test_pred).round(4) * 100
    st.markdown(f"- r^2 of training data: {train_r2}%")
    st.markdown(f"- r^2 of training data: {test_r2}%")

    st.pyplot(fig)

    return

def main():
    st.set_page_config(
        page_title="Model Complexity",
    )

    if 'complexity' not in st.session_state:
        st.session_state['complexity'] = 1
        plot_reg()

    with st.sidebar:
        complexity = st.slider(
            'How many squiggles do you want in your line?', 
            min_value=1, max_value=100, value=st.session_state.complexity, step=1, 
            key="complexity", on_change=plot_reg
        )

    return

if __name__ == '__main__':
    main()


