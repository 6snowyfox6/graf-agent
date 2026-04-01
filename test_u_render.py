import sys
import json
from pathlib import Path
sys.path.append("/home/grigoriy/graf-agent")

from plotneuralnet_renderer import PlotNeuralNetRenderer

if __name__ == "__main__":
    renderer = PlotNeuralNetRenderer(
        project_root="/home/grigoriy/graf-agent"
    )

    diagram = {
        "title": "U-Net Test",
        "layout": "u_shape",
        "nodes": [
            {"id": "input", "label": "Image Input", "kind": "input"},
            {"id": "conv1", "label": "Conv 3x3", "kind": "conv"},
            {"id": "pool1", "label": "MaxPool", "kind": "pool"},
            {"id": "conv2", "label": "Conv 5x5", "kind": "conv"},
            {"id": "pool2", "label": "MaxPool", "kind": "pool"},
            {"id": "bot", "label": "Bottleneck", "kind": "block"},
            {"id": "up2", "label": "UpConv", "kind": "conv"},
            {"id": "concat2", "label": "Concat", "kind": "concat"},
            {"id": "up1", "label": "UpConv", "kind": "conv"},
            {"id": "concat1", "label": "Concat", "kind": "concat"},
            {"id": "out", "label": "Prediction", "kind": "output"},
        ],
        "edges": [
            {"source": "input", "target": "conv1"},
            {"source": "conv1", "target": "pool1"},
            {"source": "pool1", "target": "conv2"},
            {"source": "conv2", "target": "pool2"},
            {"source": "pool2", "target": "bot"},
            {"source": "bot", "target": "up2"},
            {"source": "up2", "target": "concat2"},
            {"source": "conv2", "target": "concat2"}, # skip
            {"source": "concat2", "target": "up1"},
            {"source": "up1", "target": "concat1"},
            {"source": "conv1", "target": "concat1"}, # skip
            {"source": "concat1", "target": "out"},
        ],
    }

    try:
        res = renderer.render(diagram, output_name="test_ushape")
        print("Success!", res["pdf_path"])
    except Exception as e:
        import traceback
        traceback.print_exc()
