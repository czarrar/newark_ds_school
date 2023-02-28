import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def plot_reg():
    lr_1 = LinearRegression()

    df = pd.DataFrame([
        {"budget": 10., "revenue": 200.},
        {"budget": 20., "revenue": 5.},
        {"budget": 100., "revenue": 200.},
        {"budget": 200., "revenue": 400.},
        {"budget": 500., "revenue": 1000.},
        {"budget": 5., "revenue": st.session_state['revenue']}
    ])

    # Remove NAs
    #clean_df = df.dropna(axis=0)
    
    budget_mat = df[['budget']]
    budget = df['budget']
    revenue = df['revenue']

    lr_1.fit(budget_mat, revenue)

    lr_pred = lr_1.predict(budget_mat)

    x_max = budget.max()
    y_max = 1000.

    #plot budget and revenue
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(budget.iloc[:-1], revenue.iloc[:-1], color='b')
    plt.scatter(budget.iloc[-1:], revenue.iloc[-1:], color='r')
    plt.plot(budget, lr_pred, color='k')
    plt.xlim(-x_max * 0.05, x_max * 1.1)
    plt.ylim(-y_max * 0.05, y_max * 1.1)
    plt.title('Budget vs Revenue', fontsize = 20)
    plt.xlabel('Budget (in million)', fontsize = 15)
    plt.ylabel('Revenue (in million)', fontsize = 15)

    st.pyplot(fig)

    # Get r**2
    lr_r2 = r2_score(revenue, lr_pred).round(4) * 100
    st.markdown(f"- r^2 of regression line: {lr_r2}%")

    return


st.markdown("# Run Your Own Regression")
st.sidebar.header("Your Own Regression")
st.write(
    """This demo allows you to play with your own data and see how that effects
        the regression results. Try to create some outliers (data points that 
        are very different from the others) and see how that effects the 
        regression. Remember that the budget and revenue here are in millions. 
        So a budget of 10 would mean 10 million. Feel free to add rows and more data."""
)
st.markdown("Note: r^2 tells us how much variation our line explains of the data. It can be 0 to 100%.")

if 'revenue' not in st.session_state:
    st.session_state['revenue'] = 500.0
    plot_reg()

with st.sidebar:
    st.selectbox(
        'Select a value for the revenue of the red point',
        (0., 10., 100., 500., 1000., 5000.), index=4, on_change=plot_reg, key='revenue')


