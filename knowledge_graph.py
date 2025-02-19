import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger('veristream')

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def get_verification_status(self, fact_checks: List[Dict]) -> tuple[str, str]:
        """Determine verification status from fact checks"""
        if not fact_checks:
            return "Not Confirmed", "No fact checks available"
        
        latest_check = fact_checks[0]
        if 'claimReview' in latest_check and latest_check['claimReview']:
            review = latest_check['claimReview'][0]
            rating = review.get('textualRating', '').lower()
            
            if any(word in rating for word in ['true', 'correct', 'accurate']):
                return "Verified True", review.get('textualRating', '')
            elif any(word in rating for word in ['false', 'incorrect', 'inaccurate']):
                return "Verified False", review.get('textualRating', '')
            else:
                return "Partially Verified", review.get('textualRating', '')
        
        return "Not Confirmed", "No clear verification status"

    def add_fact(self, text: str, entities: List[Dict], fact_checks: List[Dict], sentiment: Dict):
        """Add a fact and its related information to the knowledge graph"""
        fact_id = f"fact_{hash(text)}"
        self.graph.add_node(fact_id, 
                          type='fact',
                          text=text,
                          sentiment=sentiment.get('label', 'NEUTRAL'))
        
        verification_status, verification_details = self.get_verification_status(fact_checks)
        
        verification_id = f"verification_{fact_id}"
        self.graph.add_node(verification_id,
                         type='verification',
                         status=verification_status,
                         details=verification_details)
        
        self.graph.add_edge(fact_id, verification_id, relation='verified_as')
        
        for entity in entities:
            entity_id = f"entity_{hash(entity['text'])}"
            self.graph.add_node(entity_id,
                             type='entity',
                             text=entity['text'],
                             entity_type=entity['type'])
            
            self.graph.add_edge(entity_id, fact_id, relation='mentioned_in')
            
            self.graph.add_edge(entity_id, verification_id, 
                              relation=f"verified_{verification_status.lower().replace(' ', '_')}")
        
        for i, check in enumerate(fact_checks):
            if 'claimReview' in check and check['claimReview']:
                review = check['claimReview'][0]
                check_id = f"check_{fact_id}_{i}"
                self.graph.add_node(check_id,
                                 type='fact_check',
                                 source=review.get('publisher', {}).get('name', 'Unknown'),
                                 rating=review.get('textualRating', 'Unknown'),
                                 url=review.get('url', ''))
                self.graph.add_edge(verification_id, check_id, relation='supported_by')

    def get_graph_data(self) -> Dict:
        """Return graph data in a format suitable for visualization"""
        return {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }

    def visualize_graph(self, output_file: str = 'knowledge_graph.html'):
        """Create interactive and static visualizations of the knowledge graph"""
        colors = {
            'fact': '#ff7f7f',
            'entity': '#7f7fff',
            'verification': {
                'Verified True': '#00ff00',
                'Verified False': '#ff0000',
                'Partially Verified': '#ffff00',
                'Not Confirmed': '#808080'
            },
            'fact_check': '#7fff7f'
        }

        try:
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
            net.barnes_hut()

            for node, data in self.graph.nodes(data=True):
                title = f"Type: {data['type']}<br>"
            
                if data['type'] == 'fact':
                    title += f"Text: {data['text']}<br>Sentiment: {data['sentiment']}"
                    node_color = colors['fact']
                elif data['type'] == 'entity':
                    title += f"Text: {data['text']}<br>Entity Type: {data['entity_type']}"
                    node_color = colors['entity']
                elif data['type'] == 'verification':
                    title += f"Status: {data['status']}<br>Details: {data['details']}"
                    node_color = colors['verification'].get(data['status'], '#808080')
                elif data['type'] == 'fact_check':
                    title += f"Source: {data['source']}<br>Rating: {data['rating']}"
                    if 'url' in data:
                        title += f"<br>URL: {data['url']}"
                    node_color = colors['fact_check']
            
                net.add_node(str(node), 
                            title=title,
                            color=node_color,
                            size=30 if data['type'] in ['verification', 'fact'] else 20)

            for edge in self.graph.edges(data=True):
                net.add_edge(str(edge[0]), 
                            str(edge[1]), 
                            title=edge[2].get('relation', ''),
                            physics=True)

            physics_settings = {
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 200,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0.1
                    },
                    "minVelocity": 0.75
                }
            }
            net.set_options(json.dumps(physics_settings))

            try:
                net.save_graph(output_file)
            except Exception as e:
                logger.error(f"Failed to save interactive visualization: {e}")

        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {e}")

        try:
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
            node_types = ['fact', 'entity', 'verification', 'fact_check']
            for node_type in node_types:
                if node_type == 'verification':
                    for status, color in colors['verification'].items():
                        node_list = [node for node, attr in self.graph.nodes(data=True)
                                   if attr['type'] == 'verification' and attr['status'] == status]
                        if node_list:
                            nx.draw_networkx_nodes(self.graph, pos,
                                                nodelist=node_list,
                                                node_color=color,
                                                node_size=2000,
                                                alpha=0.7)
                else:
                    node_list = [node for node, attr in self.graph.nodes(data=True)
                                if attr['type'] == node_type]
                    if node_list:
                        nx.draw_networkx_nodes(self.graph, pos,
                                            nodelist=node_list,
                                            node_color=colors[node_type],
                                            node_size=1500 if node_type == 'fact' else 1000,
                                            alpha=0.7)
        
            nx.draw_networkx_edges(self.graph, pos, 
                                 edge_color='gray', 
                                 arrows=True, 
                                 arrowsize=20,
                                 alpha=0.5)
        
            labels = {}
            for node, data in self.graph.nodes(data=True):
                if data['type'] == 'fact':
                    labels[node] = data['text'][:20] + '...'
                elif data['type'] == 'entity':
                    labels[node] = data['text']
                elif data['type'] == 'verification':
                    labels[node] = data['status']
                elif data['type'] == 'fact_check':
                    labels[node] = data['source'][:20]
        
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, label=f"Node: {label}", markersize=10)
                             for label, color in colors.items() if isinstance(color, str)]
        
            for status, color in colors['verification'].items():
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, label=f"Status: {status}",
                                                markersize=10))
        
            plt.legend(handles=legend_elements, 
                      loc='center left', 
                      bbox_to_anchor=(1, 0.5),
                      fontsize=8)
        
            plt.title("Knowledge Graph Visualization")
            plt.axis('off')
            plt.tight_layout()
        
            static_output = 'knowledge_graph_static.png'
            plt.savefig(static_output, bbox_inches='tight', dpi=300)
            plt.close()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Knowledge Graph Visualization</title>
                               <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                    .visualization {{
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 4px;
                    }}
                    .stats {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .stat-card {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 4px;
                        text-align: center;
                    }}
                    .nodes-list {{
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #ddd;
                        padding: 15px;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Knowledge Graph Visualization</h1>
                
                    <div class="visualization">
                        <h2>Interactive Visualization</h2>
                        <iframe src="{output_file}" width="100%" height="800px" frameborder="0"></iframe>
                    </div>

                    <div class="visualization">
                        <h2>Static Visualization</h2>
                        <img src="{static_output}" alt="Static Knowledge Graph">
                    </div>
                
                    <div class="stats">
                        <div class="stat-card">
                            <h3>Total Nodes</h3>
                            <p>{len(self.graph.nodes)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>Total Edges</h3>
                            <p>{len(self.graph.edges)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>Node Types</h3>
                            <p>{len(set(data['type'] for _, data in self.graph.nodes(data=True)))}</p>
                        </div>
                    </div>
                
                    <div class="nodes-list">
                        <h2>Node Details</h2>
                        <ul>
            """
        
            for node, data in self.graph.nodes(data=True):
                html_content += f"<li><strong>{node}</strong>: {str(data)}</li>"
        
            html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
            combined_output = 'knowledge_graph_combined.html'
            with open(combined_output, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            logger.error(f"Failed to create static visualization: {e}")
            raise

        return {
            'interactive': output_file,
            'static': static_output,
            'combined': combined_output
        }