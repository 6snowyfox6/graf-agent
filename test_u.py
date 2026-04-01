import graphviz
from pathlib import Path
dot = graphviz.Digraph(engine="dot")
dot.attr(rankdir='TB')  # Top to Bottom
edges = [
    ("in", "e1"), ("e1", "e2"), ("e2", "e3"), ("e3", "e4"),
    ("e4", "bot"), ("bot", "d4"),
    ("d4", "d3"), ("d3", "d2"), ("d2", "d1"), ("d1", "out"),
    ("e4", "d4"), ("e3", "d3"), ("e2", "d2"), ("e1", "d1")
]
for p in ["in", "e1", "e2", "e3", "e4", "bot", "d4", "d3", "d2", "d1", "out"]:
    dot.node(p, p)
for u, v in edges:
    if u.startswith("e") and v.startswith("d"):
        dot.edge(u, v, style="dashed", constraint="false")
    else:
        dot.edge(u, v)

# force ranks
for i in range(1, 5):
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node(f"e{i}")
        s.node(f"d{i}")

dot.render("/home/grigoriy/graf-agent/outputs/test_u", format="png")
