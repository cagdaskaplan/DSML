import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
plotly.tools.set_credentials_file(username='cagdaskaplan', api_key='iFKrLC7cS3HytpgK2RtU')
import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\stuglu\Desktop\w1.csv')


df_data_table = FF.create_table(df.head())
py.plot(df_data_table, filename='df-data-table')

trace1 = go.Scatter(
                    x=df['epoch'], y=df['weights_1'], # Data
                    mode='lines', name='1.weight' # Additional options
                   )
trace2 = go.Scatter(x=df['epoch'], y=df['weights_2'], mode='lines', name='2.weight' )
trace3 = go.Scatter(x=df['epoch'], y=df['weights_3'], mode='lines', name='3.weight')

layout = go.Layout(title='Weight alteration by epochs of a CNN',
                   plot_bgcolor='rgb(230, 230,230)')

fig = go.Figure(data=[trace1, trace2, trace3],layout=layout)

# Plot data
py.iplot(fig, filename =' CNN 2D weight alteration by cagdas',auto_open=True)