import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Transform dataset into dataframe
df = pd.read_csv("Email Analysis Dataset.csv")

# Create graph
email_graph = nx.Graph()

# Create dictionary keeping track of department comms
departments = dict()

# Iterate through each row of data, making people as nodes and communication
# between personnel as edge while increaing weight of edge
# Also storing department and calculating count of frequency of each
for index, row in df.dropna().iterrows():
    if not email_graph.has_node(row['From Name']):
        email_graph.add_node(row['From Name'], department= row['From Department'])
        
    if row['From Department'] not in departments:
        departments[row['From Department']] = 1
    else:
        count = departments[row['From Department']]
        departments[row['From Department']] = count + 1
            
    if not email_graph.has_node(row['To Name']):
        email_graph.add_node(row['To Name'], department= row['From Department'])
        
    if row['To Department'] not in departments:
        departments[row['To Department']] = 1
    else:
        count = departments[row['To Department']]
        departments[row['To Department']] = count + 1
        
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
for name, weight in top_three:
    print(f"Employee Name: {name}, Weight: {weight}")
    
# Getting departments with most sent and received communications
sorted_depts = sorted(departments.items(), key=lambda item: item[1], reverse=True)[:5]
for dept, num in sorted_depts:
    print(f"Department: {dept}, Emails Sent and Received: {num}")
    

    




