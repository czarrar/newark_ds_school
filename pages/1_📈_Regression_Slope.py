import pandas as pd
import streamlit as st

import numpy as np
#import sklearn
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


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


def manual_reg_fit(vals, coef, intercept):
    pred = intercept + vals * coef
    return pred


def plot_reg():
    st.markdown("# Best Fit Line")
    st.sidebar.header("Best Fit Line")
    st.write(
        """This demo illustrates the regression line (black) fit to our data: 
    budget (x-axis) and revenue (y-axis). Play around with the slope of your
    blue line in the side bar. Notice below how the fit of the data (r^2) changes 
    as the slope changes. Can you get a better fit to the  then the regression?
    Enjoy!"""
    )
    st.markdown("Note: r^2 tells us how much variation our line explains of the data. It can be 0 to 100%.")

    dat, X_train, X_test, y_train, y_test = load_data()

    budget_mat = X_train[['budget']]/1e8
    budget = X_train['budget']/1e8
    revenue = y_train['revenue']/1e8

    lr_1 = LinearRegression()
    lr_1.fit(budget_mat, revenue)

    x_max = budget.max()
    y_max = revenue.max()

    lr_pred = lr_1.predict(budget_mat)
    my_pred = manual_reg_fit(budget, st.session_state.slope, lr_1.intercept_)

    #plot budget and revenue
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(budget, revenue)
    plt.plot(budget, lr_pred, color='k')
    plt.plot(budget, my_pred, linestyle='dashed', color='b')
    plt.xlim(-x_max * 0.05, x_max * 1.1)
    plt.ylim(-y_max * 0.05, y_max * 1.1)
    plt.title('Budget vs Revenue', fontsize = 20)
    plt.xlabel('Budget (in million)', fontsize = 15)
    plt.ylabel('Revenue (in million)', fontsize = 15)

    st.pyplot(fig)

    # Get r**2
    lr_r2 = r2_score(revenue, lr_pred).round(4) * 100
    my_r2 = r2_score(revenue, my_pred).round(4) * 100
    st.markdown(f"- r^2 of regression line: {lr_r2}%")
    st.markdown(f"- r^2 of my line: {my_r2}%")

    return


def main():
    st.set_page_config(
        page_title="DS Demo",
    )

    if 'slope' not in st.session_state:
        st.session_state['slope'] = 1.0
        plot_reg()

    with st.sidebar:
        slope = st.slider(
            'What slope do you want to use?', 
            min_value=0.0, max_value=20.0, value=st.session_state.slope, step=0.5, 
            key="slope", on_change=plot_reg
        )

    return

if __name__ == '__main__':
    main()
