import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Transform dataset into dataframe
df = pd.read_csv("Email Analysis Dataset.csv")

# Create graph
email_graph = nx.Graph()

# Iterate through each row of data, making people as nodes and communication
# between personnel as edge while increaing weight of edge
for index, row in df.dropna().iterrows():
    if not email_graph.has_node(row['From Name']):
        email_graph.add_node(row['From Name'])
    if not email_graph.has_node(row['To Name']):
        email_graph.add_node(row['To Name'])
        
    if email_graph.has_edge(row['From Name'], row['To Name']):
        # Increment the weight
        email_graph[row['From Name']][row['To Name']]['weight'] += 1
    else:
        # Create the edge with initial weight = 1
        email_graph.add_edge(row['From Name'], row['To Name'], weight=1)

# Drawing the graph
# nx.draw(email_graph, with_labels = True, node_color="blue", font_weight="bold")
# plt.show()

# Getting top three personnel with most communication
top_three = sorted(email_graph.degree, key=lambda x: x[1], reverse=True)[:3]
print(top_three)
    

    




