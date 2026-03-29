# Skill: plotneuralnet_label_normalization

## Goals
- keep labels short
- preserve architecture meaning
- never over-collapse semantics

## Safe replacements
- Initial Convolutional Layers -> Conv Stem
- CNN Layers -> CNN Blocks
- Transformer Encoder Blocks -> Transformer Blocks
- Feature Fusion Module -> Fusion
- Classification Head -> Cls Head
- Output Layer -> Output

## Do not do
- do not replace every unfamiliar label with Block
- do not force 1-2 word labels if meaning is lost