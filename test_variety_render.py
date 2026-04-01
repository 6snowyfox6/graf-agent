import sys
import os
from pathlib import Path

sys.path.append("/home/grigoriy/graf-agent")

from plotneuralnet_renderer import PlotNeuralNetRenderer

def generate_diagram(renderer, diagram, name):
    try:
        res = renderer.render(diagram, output_name=name)
        pdf_path = res["pdf_path"]
        png_path = Path(pdf_path).with_suffix(".png")
        os.system(f"pdftocairo -png {pdf_path} {png_path.with_name(name)} && mv {png_path.with_name(name)}-1.png {png_path}")
        print(f"OK: {name} → {png_path}")
        return str(png_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL: {name}: {e}")
        return None

if __name__ == "__main__":
    renderer = PlotNeuralNetRenderer(project_root="/home/grigoriy/graf-agent")

    # ──────────────── 1. ResNet (Linear, skip-connection) ────────────────
    resnet = {
        "title": "ResNet Block",
        "layout": "linear",
        "nodes": [
            {"id": "in", "label": "Input", "kind": "input"},
            {"id": "c1", "label": "Conv1", "kind": "conv"},
            {"id": "c2", "label": "Conv2", "kind": "conv"},
            {"id": "add1", "label": "Add", "kind": "sum"},
            {"id": "c3", "label": "Conv3", "kind": "conv"},
            {"id": "out", "label": "Output", "kind": "output"},
        ],
        "edges": [
            {"source": "in", "target": "c1"},
            {"source": "c1", "target": "c2"},
            {"source": "c2", "target": "add1"},
            {"source": "c1", "target": "add1"},   # skip
            {"source": "add1", "target": "c3"},
            {"source": "c3", "target": "out"},
        ]
    }

    # ──────────────── 2. U-Net (U-shape, skip-connections) ────────────────
    unet = {
        "title": "U-Net",
        "layout": "u_shape",
        "nodes": [
            {"id": "e1", "label": "Enc1", "kind": "conv"},
            {"id": "p1", "label": "Pool", "kind": "pool"},
            {"id": "e2", "label": "Enc2", "kind": "conv"},
            {"id": "bot", "label": "Bottleneck", "kind": "block"},
            {"id": "u2", "label": "Up2", "kind": "conv"},
            {"id": "cat2", "label": "Cat", "kind": "concat"},
            {"id": "u1", "label": "Up1", "kind": "conv"},
            {"id": "cat1", "label": "Cat", "kind": "concat"},
            {"id": "out", "label": "Output", "kind": "output"},
        ],
        "edges": [
            {"source": "e1", "target": "p1"},
            {"source": "p1", "target": "e2"},
            {"source": "e2", "target": "bot"},
            {"source": "bot", "target": "u2"},
            {"source": "u2", "target": "cat2"},
            {"source": "e2", "target": "cat2"},   # skip
            {"source": "cat2", "target": "u1"},
            {"source": "u1", "target": "cat1"},
            {"source": "e1", "target": "cat1"},   # skip
            {"source": "cat1", "target": "out"},
        ]
    }

    # ──────────────── 3. Inception (Branching) ────────────────
    inception = {
        "title": "Inception Module",
        "layout": "linear",
        "nodes": [
            {"id": "in", "label": "Input", "kind": "input"},
            {"id": "b1", "label": "1x1 Conv", "kind": "conv"},
            {"id": "b2", "label": "3x3 Conv", "kind": "conv"},
            {"id": "b3", "label": "5x5 Conv", "kind": "conv"},
            {"id": "cat", "label": "Concat", "kind": "concat"},
            {"id": "out", "label": "Output", "kind": "output"},
        ],
        "edges": [
            {"source": "in", "target": "b1"},
            {"source": "in", "target": "b2"},
            {"source": "in", "target": "b3"},
            {"source": "b1", "target": "cat"},
            {"source": "b2", "target": "cat"},
            {"source": "b3", "target": "cat"},
            {"source": "cat", "target": "out"},
        ]
    }

    # ──────────────── 4. Autoencoder (U-shape, no skips) ────────────────
    autoencoder = {
        "title": "Autoencoder",
        "layout": "u_shape",
        "nodes": [
            {"id": "in", "label": "Input", "kind": "input"},
            {"id": "enc1", "label": "Enc1", "kind": "conv"},
            {"id": "enc2", "label": "Enc2", "kind": "conv"},
            {"id": "z", "label": "Latent Z", "kind": "fc"},
            {"id": "dec1", "label": "Dec1", "kind": "conv"},
            {"id": "dec2", "label": "Dec2", "kind": "conv"},
            {"id": "out", "label": "Reconstruction", "kind": "output"},
        ],
        "edges": [
            {"source": "in", "target": "enc1"},
            {"source": "enc1", "target": "enc2"},
            {"source": "enc2", "target": "z"},
            {"source": "z", "target": "dec1"},
            {"source": "dec1", "target": "dec2"},
            {"source": "dec2", "target": "out"},
        ]
    }

    results = {}
    for name, diagram in [
        ("walkthrough_resnet", resnet),
        ("walkthrough_unet", unet),
        ("walkthrough_inception", inception),
        ("walkthrough_autoencoder", autoencoder),
    ]:
        results[name] = generate_diagram(renderer, diagram, name)

    print("\n── Summary ──")
    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}: {path}")
