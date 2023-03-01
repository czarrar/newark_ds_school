# Code borrowed from: https://www.kaggle.com/code/markow/validation-set-approach-and-polynomial-regression

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

@st.cache_data
def load_data():
    with st.spinner("Loading data..."):
        #grid needed for plotting charts: 101 equidistant points between -20 and 20
        grid = np.linspace(-20,20,101) 
        #artificial input values x match the grid
        x = np.linspace(-20,20,101) 
        #artifical output values y: polynomial of degree 5 plus a random error term (normally distributed with mean 0 and standard deviation 2, using a fixed seed for reproducible results)
        #So the best estimator for y should be a polynom of degree 5
        np.random.seed(111)
        y = 0.000005*x**5 + 2*np.random.randn(len(x))

        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.4)

    return x, y, X_train, X_test, y_train, y_test


def plot_reg():

    st.markdown("# Model Complexity")
    st.sidebar.header("Model Complexity")
    st.markdown(
        """The slider on the left will determine the number of squiggles in the line.
        More squiggles means a more complex model that fits the training data better.
        Are more squiggles always better? 
        Try to see if you can find the number of squiggles that gives the best answer."""
    )
    st.markdown("""Note: r^2 tells us how much variation our line explains of the data. 
It can be 0 to 100%. It can also be negative if the model explains the data worse than a horizontal line.""")


    # Get the data
    x, y, x_train, x_test, y_train, y_test = load_data()

    # Gather data
    plot_df_train = pd.DataFrame({'x': x_train, 'y': y_train})
    plot_df_train = plot_df_train.sort_values('x').reset_index(drop=True)

    plot_df_test = pd.DataFrame({'x': x_test, 'y': y_test})
    plot_df_test = plot_df_test.sort_values('x').reset_index(drop=True)

    # Fit the data
    #model = make_pipeline(PolynomialFeatures(int(st.session_state['complexity'])), LinearRegression())
    model = make_pipeline(
        SplineTransformer(n_knots=int(st.session_state['complexity']) + 2, degree =int(st.session_state['complexity'])), 
        LinearRegression()
    )
    
    model.fit(plot_df_train[['x']], plot_df_train.y)

    plot_df_train['ypred'] = model.predict(plot_df_train[['x']])
    plot_df_test['ypred'] = model.predict(plot_df_test[['x']])


    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    xmin = x.min() - np.abs(x.min()) * 0.2
    xmax = x.max() + np.abs(x.max()) * 0.2
    ymin = y.min() - np.abs(y.min()) * 0.2
    ymax = y.max() + np.abs(y.max()) * 0.2

    ax1.title.set_text('Training Data')
    ax1.scatter(plot_df_train.x, plot_df_train.y)
    ax1.plot(plot_df_train.x, plot_df_train.ypred, color='r')
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])

    ax2.title.set_text('Test Data')
    ax2.scatter(plot_df_test.x, plot_df_test.y)
    ax2.plot(plot_df_test.x, plot_df_test.ypred, color='r')
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])

    # Evaluate model
    train_r2 = r2_score(plot_df_train.y, plot_df_train.ypred).round(4) * 100
    test_r2  = r2_score(plot_df_test.y, plot_df_test.ypred).round(4) * 100
    st.markdown(f"- r^2 of training data: {train_r2:.2f}%")
    st.markdown(f"- r^2 of testing data: {test_r2:.2f}%")

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
            min_value=0, max_value=30, value=st.session_state.complexity, step=1, 
            key="complexity", on_change=plot_reg
        )

    return

if __name__ == '__main__':
    main()


