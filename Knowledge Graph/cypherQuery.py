def generate_cypher_query(processed_statement):
    organizations = processed_statement["Organizations"]
    fact = processed_statement["Fact"]
    truth_value = processed_statement["Truth Value"]
    nodes = processed_statement["Knowledge Graph Nodes"]
    edges = processed_statement["Knowledge Graph Edges"]

    query = []

    for node, attributes in nodes:
        if attributes['label'] == 'Organization':
            query.append(f'CREATE ({node.replace(" ", "_")}:Organization {{name: "{node}"}})')
        elif attributes['label'] == 'Fact':
            query.append(f'CREATE ({node.replace(" ", "_")}:Fact {{description: "{node}"}})')

    for source, target, attributes in edges:
        source_id = source.replace(" ", "_")
        target_id = target.replace(" ", "_")
        relation = attributes["relation"]
        query.append(f'CREATE ({source_id})-[:RELATION {{truth_value: "{relation}"}}]->({target_id})')

    return "\n".join(query)


processed_statement = {
    "Organizations": ["XYZ Corporation", "ABC Organization"],
    "Fact": "XYZ Corporation acquired ABC Organization",
    "Truth Value": "false",
    "Knowledge Graph Nodes": [
        ("XYZ Corporation", {"label": "Organization"}),
        ("ABC Organization", {"label": "Organization"}),
        ("XYZ Corporation acquired ABC Organization", {"label": "Fact"})
    ],
    "Knowledge Graph Edges": [
        ("XYZ Corporation", "XYZ Corporation acquired ABC Organization", {"relation": "false"}),
        ("ABC Organization", "XYZ Corporation acquired ABC Organization", {"relation": "false"})
    ]
}

cypher_query = generate_cypher_query(processed_statement)
print(cypher_query)
