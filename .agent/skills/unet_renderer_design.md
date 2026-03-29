# Skill: unet_renderer_design

Generic linear PlotNeuralNet renderer is not enough for article-style U-Net layouts.

## Required features
- explicit encoder branch
- explicit decoder branch
- skip connections
- controlled symmetric layout
- separate colors for encoder/decoder/attention
- optional legend

## Recommendation
Implement a dedicated U-Net renderer instead of overloading the generic linear renderer.