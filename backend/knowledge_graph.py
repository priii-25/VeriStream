# backend/knowledge_graph.py
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

    def get_verification_status(self, fact_checks: List[Dict]) -> Tuple[str, str]:
        """Determine verification status from fact checks."""
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
        """Add a fact and its related information to the knowledge graph."""
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
        """Return graph data in a format suitable for visualization."""
        return {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }

    def visualize_graph(self, output_file: str = 'knowledge_graph.html'):
        """Create a self-contained interactive visualization of the knowledge graph."""
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
            # Initialize PyVis network
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
            net.barnes_hut()

            # Add nodes and edges
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
                        title += f"<br>URL: <a href='{data['url']}' target='_blank'>{data['url']}</a>"
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

            # Physics settings
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

            # Generate self-contained HTML
            net.write_html(output_file, notebook=False)  # Ensures all JS is embedded
            logger.info(f"Self-contained interactive knowledge graph saved to {output_file}")

            # Static visualization (unchanged)
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
            
            labels = {node: f"{data['text'][:20]}..." if data['type'] == 'fact' else data.get('text', node) 
                     for node, data in self.graph.nodes(data=True)}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            static_output = 'knowledge_graph_static.png'
            plt.title("Knowledge Graph Visualization")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(static_output, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Static knowledge graph saved to {static_output}")

        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise

        return output_file

if __name__ == "__main__":
    kg = KnowledgeGraphManager()
    kg.add_fact(
        text="The Earth is flat",
        entities=[{"text": "Earth", "type": "LOCATION"}],
        fact_checks=[{"claimReview": [{"textualRating": "False", "publisher": {"name": "Science Daily"}}]}],
        sentiment={"label": "NEGATIVE"}
    )
    kg.visualize_graph()