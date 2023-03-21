import pickle
import plotly.graph_objects as go
import sys
iteration = sys.argv[1]
plot_data = pickle.load(open('results/1st_order_3D/0/images/primitive_0_iter_%s.pickle' % iteration, 'rb'))
fig = go.Figure(data=plot_data['3D_plot'])
fig.show()
