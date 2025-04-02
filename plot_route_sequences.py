import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Create output directory
os.makedirs('route_visualizations', exist_ok=True)

# Read the route data
df = pd.read_csv('bus_route_source.csv')

# Dictionary to store route information
route_info = defaultdict(list)

# Process each route
for route_id, group in df.groupby('#Id'):
    route_name = group['Route Name'].iloc[0]
    route_number = str(group['Route Number'].iloc[0]).strip()  # Convert to string and strip whitespace
    direction = group['Route Direction'].iloc[0]
    
    # Sort stops by Route Order
    stops = group.sort_values('Route Order')
    
    # Create a graph for this route
    G = nx.DiGraph()
    
    # Add nodes and edges for this route
    for i in range(len(stops) - 1):
        current_stop = stops.iloc[i]
        next_stop = stops.iloc[i + 1]
        
        # Add nodes if they don't exist
        G.add_node(current_stop['Bus Stop Name'], 
                  route_number=route_number,
                  direction=direction,
                  order=current_stop['Route Order'])
        G.add_node(next_stop['Bus Stop Name'],
                  route_number=route_number,
                  direction=direction,
                  order=next_stop['Route Order'])
        
        # Add edge with route information
        G.add_edge(current_stop['Bus Stop Name'],
                  next_stop['Bus Stop Name'],
                  route_number=route_number,
                  direction=direction,
                  route_name=route_name)
    
    # Store route information
    route_info[route_number].append({
        'name': route_name,
        'direction': direction,
        'stops': stops['Bus Stop Name'].tolist()
    })
    
    # Create visualization for this route
    plt.figure(figsize=(15, 10))
    
    # Use hierarchical layout for better visualization of route sequence
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add title with route information
    plt.title(f"Route {route_number}: {route_name}\nDirection: {direction}", pad=20, fontsize=14)
    
    # Save the plot
    safe_route_number = "".join(c for c in route_number if c.isalnum())  # Remove special characters for filename
    plt.savefig(f'route_visualizations/route_{safe_route_number}_{direction}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Create a text summary of routes
with open('route_summary.txt', 'w') as f:
    f.write("Chennai Bus Route Summary\n")
    f.write("=======================\n\n")
    
    for route_number, routes in route_info.items():
        f.write(f"\nRoute {route_number}:\n")
        f.write("-" * (len(route_number) + 7) + "\n")
        
        for route in routes:
            f.write(f"\nDirection: {route['direction']}\n")
            f.write(f"Name: {route['name']}\n")
            f.write("Stops:\n")
            for i, stop in enumerate(route['stops'], 1):
                f.write(f"  {i}. {stop}\n")
            f.write("\n")

print("Individual route visualizations saved in 'route_visualizations' directory")
print("Detailed route summary saved as 'route_summary.txt'") 