import re
import networkx as nx
from transformers import pipeline

class VeriStream:
    def __init__(self):
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.graph = nx.DiGraph()

    def extract_statement(self, text):
        match = re.match(r"^(.*) is (true|false)$", text.strip(), re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            truth_value = match.group(2).lower()
            return fact, truth_value
        return None, None

    def extract_organizations(self, text):
        ner_results = self.ner_pipeline(text)
        organizations = []
        current_entity = []

        for result in ner_results:
            if result["entity"].endswith("ORG"):
                token = result["word"]
                if token.startswith("##"):
                    current_entity[-1] += token[2:]
                else:
                    if current_entity:
                        organizations.append(" ".join(current_entity))
                    current_entity = [token]
            elif current_entity:
                organizations.append(" ".join(current_entity))
                current_entity = []

        if current_entity:
            organizations.append(" ".join(current_entity))

        organizations = list(dict.fromkeys(organizations))
        return organizations

    def update_knowledge_graph(self, organizations, fact, truth_value):
        """Add nodes and edges to the knowledge graph."""
        for org in organizations:
            self.graph.add_node(org, label="Organization")
            self.graph.add_node(fact, label="Fact")
            self.graph.add_edge(org, fact, relation=truth_value)

    def process_statement(self, text):
        """Process the statement to extract, analyze, and build the graph."""
        fact, truth_value = self.extract_statement(text)
        if not fact or not truth_value:
            return {"error": "Invalid statement format"}
        
        organizations = self.extract_organizations(fact)
        if not organizations:
            return {"error": "No organizations found in fact"}

        self.update_knowledge_graph(organizations, fact, truth_value)
        return {
            "organizations": organizations,
            "fact": fact,
            "truth_value": truth_value,
            "graph": {
                "nodes": list(self.graph.nodes(data=True)),
                "edges": list(self.graph.edges(data=True))
            }
        }

if __name__ == "__main__":
    veristream = VeriStream()
    statement = "XYZ Corporation acquired ABC Organization is false"
    result = veristream.process_statement(statement)

    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Processed Statement:")
        print("Organizations:", ", ".join(result["organizations"]))
        print("Fact:", result["fact"])
        print("Truth Value:", result["truth_value"])
        print("Knowledge Graph Nodes:", result["graph"]["nodes"])
        print("Knowledge Graph Edges:", result["graph"]["edges"])