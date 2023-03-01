import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the DS Linear Regression Demos! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ## Prior Demos
    We've already gone through the following lessons:
    - [Week 2 - Basics of Scripting and Data Frames](https://colab.research.google.com/drive/1-Mcz3MqlRsQziBpP47lT6ni930Q_hjo8)
    - [Week 3 - Basics of Visualization](https://colab.research.google.com/drive/1_hPe-gIaIFDyK45-0tsR8QAzSvRJWqKo#scrollTo=dlUqUPq57r5q)
    - [Week 3 - Data Visualization & Exploratory Data Analysis](https://colab.research.google.com/drive/1YNVq-vEQl5hJjKWicQeJKH4-LGk8TcFM#scrollTo=O-4DHSWBF1tU)
    - [Week 4 - Data Merging/Joining](https://colab.research.google.com/drive/1ElK7m8pwRi3qCANOaQUHOGFg0-s0FX67#scrollTo=SZJtHBwk9IBT)
    - [Week 5 - Linear Regression](https://colab.research.google.com/drive/1gihoB9Aw2gQmGE5nvOwB-Zrhla7MSFN8?authuser=1#scrollTo=4e2la_R8tUpu)

    ## Todays Class
    - Go through some demos on linear regression. Here are some questions to consider:
        - How can this linear regression be used to make predictions?
        - How does linear regression give us the best fit line? 
        - Where might linear regression do poorly? 
        - When should we use simpler models like linear regression versus more complex models that more accurately fit the data?
    - Overview of more 'complex' models like decision trees and neural networks. Discuss regression vs classification problems.
    - Go through some cool demos!
"""
)