#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
import numpy as np


# In[4]:


st.write("""
# My first app
## Hello *world!*
""")


# In[7]:


df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

st.table(df)

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

st.write("first twenty countries who reported COVID 19")

covid19 = pd.read_csv("https://covid19.who.int/WHO-COVID-19-global-data.csv")

d=covid19.tail(20)
d

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
# In[ ]:

# import gmplot package
import gmplot

latitude_list = [30.3358376, 30.307977, 30.3216419]
longitude_list = [77.8701919, 78.048457, 78.0413095]

gmap3 = gmplot.GoogleMapPlotter(30.3164945,
                                78.03219179999999, 13)

# scatter method of map object
# scatter points on the google map
gmap3.scatter(latitude_list, longitude_list, '# FF0000',
              size=40, marker=False)

# Plot method Draw a line in
# between given coordinates
gmap3.plot(latitude_list, longitude_list,
           'cornflowerblue', edge_width=2.5)

streamlit