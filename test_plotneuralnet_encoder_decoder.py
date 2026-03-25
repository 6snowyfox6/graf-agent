from plotneuralnet_renderer import PlotNeuralNetRenderer

diagram = {
    "title": "Encoder Decoder",
    "nodes": [
        {"id": "input", "label": "Input 256x256x3", "kind": "input"},
        {"id": "enc1", "label": "Encoder 64", "kind": "block"},
        {"id": "pool1", "label": "MaxPool", "kind": "pool"},
        {"id": "enc2", "label": "Encoder 128", "kind": "block"},
        {"id": "pool2", "label": "MaxPool", "kind": "pool"},
        {"id": "bottleneck", "label": "Bottleneck 256", "kind": "block"},
        {"id": "dec1", "label": "Decoder 128", "kind": "block"},
        {"id": "dec2", "label": "Decoder 64", "kind": "block"},
        {"id": "output", "label": "Output Mask", "kind": "output"},
    ],
    "edges": [
        {"source": "input", "target": "enc1"},
        {"source": "enc1", "target": "pool1"},
        {"source": "pool1", "target": "enc2"},
        {"source": "enc2", "target": "pool2"},
        {"source": "pool2", "target": "bottleneck"},
        {"source": "bottleneck", "target": "dec1"},
        {"source": "dec1", "target": "dec2"},
        {"source": "dec2", "target": "output"},
    ],
}

renderer = PlotNeuralNetRenderer(project_root=".")
result = renderer.render(diagram, output_name="encoder_decoder")
print(result)
