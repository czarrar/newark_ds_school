import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def plot_reg(edited_df):
    lr_1 = LinearRegression()

    # Remove NAs
    clean_df = edited_df.dropna(axis=0)
    
    budget_mat = clean_df[['budget']]
    budget = clean_df['budget']
    revenue = clean_df['revenue']

    lr_1.fit(budget_mat, revenue)

    lr_pred = lr_1.predict(budget_mat)

    x_max = budget.max()
    y_max = revenue.max()

    #plot budget and revenue
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(budget, revenue)
    plt.plot(budget, lr_pred, color='k')
    plt.xlim(-x_max * 0.05, x_max * 1.1)
    plt.ylim(-y_max * 0.05, y_max * 1.1)
    plt.title('Budget vs Revenue', fontsize = 20)
    plt.xlabel('Budget (in million)', fontsize = 15)
    plt.ylabel('Revenue (in million)', fontsize = 15)

    st.pyplot(fig)

    # Get r**2
    lr_r2 = r2_score(revenue, lr_pred).round(4) * 100
    st.markdown(f"- r^2 (percent fit): {lr_r2}%")

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
st.markdown("""Note: r^2 tells us how much variation our line explains of the data. 
It can be 0 to 100%. It can also be negative if the model explains the data worse than a horizontal line.""")

df = pd.DataFrame(
    [
       {"budget": 10, "revenue": 100},
       {"budget": 20, "revenue": 5},
       {"budget": 100, "revenue": 50},
       {"budget": 200, "revenue": 400},
   ]
)

with st.sidebar:
    edited_df = st.experimental_data_editor(df, num_rows="dynamic")

    run = st.button('Run')

if run:
    plot_reg(edited_df)


