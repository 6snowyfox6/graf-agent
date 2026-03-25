from plotneuralnet_renderer import PlotNeuralNetRenderer

diagram = {
    "title": "Simple CNN",
    "nodes": [
        {"id": "input", "label": "Input 224x224x3", "kind": "input"},
        {"id": "conv1", "label": "Conv 64", "kind": "conv"},
        {"id": "pool1", "label": "MaxPool", "kind": "pool"},
        {"id": "fc1", "label": "FC 256", "kind": "fc"},
        {"id": "out", "label": "Classifier", "kind": "output"},
    ],
    "edges": [
        {"source": "input", "target": "conv1"},
        {"source": "conv1", "target": "pool1"},
        {"source": "pool1", "target": "fc1"},
        {"source": "fc1", "target": "out"},
    ],
}

renderer = PlotNeuralNetRenderer(project_root=".")
result = renderer.render(diagram, output_name="simple_cnn")
print(result)
