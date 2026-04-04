---
description: Agent-to-agent execution handoff log
---

# Handoff Log

## Handoff Format (standard)
- Context: what task and target quality level.
- Changes: exact files and key logic updates.
- Checks: commands/tests and output artifacts.
- Risks: what still looks unstable or fragile.
- Next: one concrete next action for continuation.

## 2026-04-04 - Python-first generation path for PlotNeuralNet

### Context
- Goal: shift model architecture rendering toward PlotNeuralNet Python-script generation per diagram, with safe fallback.
- User direction: continue implementation flow without stop, keep rollback option.

### Changes
- `plotneuralnet_renderer.py`
  - Added generic Python backend path:
    - `render()` now supports `python_backend` mode (`auto` by default).
    - canonical U-Net still uses dedicated Python generator.
    - non-canonical diagrams use generated `*_gen.py` scripts that call `pycore.tikzeng`.
    - if python generation fails, renderer falls back to previous direct-TeX path and stores warning.
- `main.py`
  - Strengthened system prompts so `python_backend` is treated as protected pipeline field.
  - JSON format template now includes `"python_backend": "on"` for `layout_hint=model_architecture`.
  - Generator instructions now explicitly request `python_backend: on` for model architectures.
- `diagram_types/model_architecture.json`
  - Added explicit extra rule to always include `python_backend: on`.

### Checks
- `python3 -m py_compile plotneuralnet_renderer.py`
- `python3 test_variety_render.py`
- `python3 test_u_render.py`
- Verified generated scripts in outputs:
  - `outputs/walkthrough_unet/walkthrough_unet_gen.py`
  - `outputs/walkthrough_resnet/walkthrough_resnet_gen.py`
  - `outputs/walkthrough_inception/walkthrough_inception_gen.py`
  - `outputs/walkthrough_autoencoder/walkthrough_autoencoder_gen.py`

### Risks
- Generic Python path currently wraps precomputed TikZ fragments (not fully macro-level decomposition like handcrafted PlotNeuralNet examples).
- If target architecture needs strict example-grade styling (e.g., canonical ResNet paper style), architecture-specific Python generators are still preferable.

### Next
1. Add first dedicated Python generator for ResNet family (`render_contract=resnet_canonical`) using `blocks.py`-style patterns.
2. Keep generic path as fallback for unknown architectures.

## 2026-04-04 - Dedicated ResNet python generator (auto-detect enabled)

### Context
- Continued Python-backend migration with focus on reference-like ResNet visuals.
- Requirement: keep uninterrupted progress and preserve safe rollback.

### Changes
- `plotneuralnet_renderer.py`
  - Added `canonical_resnet` route:
    - `render_contract == canonical_resnet` now triggers dedicated Python script generation.
  - Added auto-detection:
    - if `python_backend` is enabled and diagram text looks like ResNet, dedicated ResNet generator is used automatically.
  - New helpers:
    - `_looks_like_resnet`
    - `_infer_resnet_depth` (18/34 explicit; 50/101/152 currently mapped to 34-style visual template with warning)
    - `_render_canonical_resnet_via_python`
    - `_build_resnet_python_script`
  - ResNet script now emits stable PlotNeuralNet primitives (`to_Conv`, `to_ConvRes`, `to_Pool`, `to_SoftMax`, `to_connection`) and writes `*_gen.py`.

### Checks
- `python3 -m py_compile plotneuralnet_renderer.py`
- `python3 test_variety_render.py`
- `python3 test_u_render.py`
- Result: all walkthrough renders are successful, including `walkthrough_resnet`.

### Risks
- ResNet-50/101/152 are not yet true bottleneck-style visuals; currently rendered as 34-like block pattern to keep stability.
- Final artistic parity with the user reference still needs per-architecture fine-tuning of spacing/opacity/captions.

### Next
1. Add true bottleneck block visual mode for ResNet-50+.
2. Tune spacing/width curves for compact “paper style” across ResNet18/34.

## 2026-04-04 - ResNet bottleneck mode implemented for 50/101/152

### Context
- User requested continuation without stops and explicit move from 34-like approximation to real bottleneck appearance for deep ResNet variants.

### Changes
- `plotneuralnet_renderer.py`
  - `_infer_resnet_depth` now returns true detected depth (`18/34/50/101/152`) without remapping deep variants to 34.
  - `_build_resnet_python_script` split into two modes:
    - basic block mode for `ResNet-18/34` (existing `to_ConvRes` style).
    - bottleneck mode for `ResNet-50/101/152`:
      - per block: `1x1 reduce -> 3x3 -> 1x1 expand -> Sum`
      - residual shortcut is explicit:
        - stage first block uses projection branch (`Proj`)
        - other blocks use identity branch to `Sum`
      - stage counts now follow canonical depths:
        - 50: `[3,4,6,3]`
        - 101: `[3,4,23,3]`
        - 152: `[3,8,36,3]`

### Checks
- `python3 -m py_compile plotneuralnet_renderer.py`
- `python3 test_variety_render.py` (all walkthrough renders successful)
- Canonical ResNet renders:
  - `outputs/review_resnet18_bneck/review_resnet18_bneck.pdf`
  - `outputs/review_resnet50_bneck/review_resnet50_bneck.pdf`
  - `outputs/review_resnet101_bneck/review_resnet101_bneck.pdf`
  - `outputs/review_resnet152_bneck/review_resnet152_bneck.pdf`

### Risks
- Very deep variants (especially 152) can become visually dense due to exact stage depth.
- Next visual polish step is layout compression/grouping for readability while preserving bottleneck semantics.

### Next
1. Add stage-wise visual compression for 101/152 (optional collapsed repeats with explicit `xN` labels).
2. Tune bottleneck spacing/opacity to match user reference style more tightly.

## 2026-04-04 - Prompt-driven template router in main pipeline

### Context
- User asked to continue toward “prompt modifies same/similar python architecture file” behavior.
- Goal: make template selection deterministic before rendering, not implicit only in renderer internals.

### Changes
- `main.py`
  - Added `infer_model_template(diagram, user_task)`:
    - detects template family by prompt/title/nodes text.
    - current families: `unet`, `resnet`, `inception`, `gan`, `autoencoder`, `transformer`, fallback `generic_model`.
  - Added `apply_model_template_router(diagram, user_task)`:
    - enforces model rendering stack fields:
      - `layout_hint = model_architecture`
      - `renderer = plotneuralnet`
      - `python_backend = on`
    - sets/keeps `render_contract` and `template_id`.
    - normalizes layout by family (`u_shape` for U-Net, `linear` for ResNet).
  - Integrated router into:
    - `generate_diagram()` for `layout_hint=model_architecture`.
    - `improve_diagram()` post-processing path (so critique/improve no longer drops routing fields).
  - Prompts in critique/improve now treat `render_contract` + `template_id` as protected system fields.
  - Fixed a pipeline bug:
    - `improve_diagram()` previously always applied `normalize_general_diagram()`; now only for `general` diagrams, preventing kind corruption for model architectures.

### Checks
- `python3 -m py_compile main.py plotneuralnet_renderer.py`
- `python3 test_variety_render.py` (all walkthrough outputs successful)
- Router smoke test:
  - `ResNet-50` -> `template_id=resnet`, `render_contract=canonical_resnet`
  - `U-Net` -> `template_id=unet`, `render_contract=canonical_unet`
  - `GAN` -> `template_id=gan`, `render_contract=generic_python_template`

### Risks
- Non-canonical families (`inception`, `gan`, `transformer`, etc.) still route to generic python-template contract until dedicated generators are implemented.

### Next
1. Implement dedicated Python generators for `inception` and `gan`.
2. Add collapsed-stage rendering for deep ResNet (101/152) to keep visuals compact.

## 2026-04-04 - Added canonical YOLO template

### Context
- User asked to add a dedicated YOLO template, not generic fallback.

### Changes
- `plotneuralnet_renderer.py`
  - Added `canonical_yolo` contract path in `render()`.
  - Added auto-detection for YOLO in python backend mode (`_looks_like_yolo`).
  - Added YOLO variant detection (`_infer_yolo_variant` for v3/v5/v8).
  - Added dedicated generator:
    - `_render_canonical_yolo_via_python`
    - `_build_yolo_python_script`
  - YOLO script includes canonical pipeline blocks:
    - Backbone
    - Neck (upsample/fusion)
    - Multi-scale heads (S/M/L)
    - Detection fusion/output node
- `main.py`
  - Template router now maps any YOLO prompt to:
    - `template_id = yolo`
    - `render_contract = canonical_yolo`
    - `layout = linear`

### Checks
- `python3 -m py_compile main.py plotneuralnet_renderer.py`
- Live end-to-end run with local models:
  - prompt: YOLOv8 architecture
  - pipeline: generate -> critique -> improve -> render
  - result:
    - `template_id = yolo`
    - `render_contract = canonical_yolo`
    - output:
      - `outputs/demo_yolo_live_ascii/demo_yolo_live_ascii.pdf`
      - `outputs/demo_yolo_live_ascii/demo_yolo_live_ascii.png`

### Risks
- Current YOLO template is canonicalized and visually consistent, but not yet variant-perfect for all families (v3/v5/v8 specific block-level differences can be expanded later).

### Next
1. Add variant-specific topology presets (`v3` anchors/FPN, `v5` C3, `v8` C2f decoupled head).
2. Add style controls in JSON (`template_style=paper|minimal`) for reference matching.

## 2026-04-04 - PlotNeuralNet style and renderer quality pass

### Scope
- Goal: make model diagrams closer to PlotNeuralNet example quality and user reference style.
- Main work areas:
  - global PlotNeuralNet styling
  - linear reference-style ResNet rendering
  - branching (Inception) routing cleanup
  - U-Net routing and skip-lane behavior

### Files changed
- `plotneuralnet_renderer.py`
- `.agent/workflows/create_model_diagram.md`
- `.agent/workflows/improve_plotneuralnet_output.md`
- `.agent/workflows/handoff_log.md` (this file)

### Renderer-level changes
- Added support for explicit node-level macro override via JSON:
  - `node.macro`
  - `node.params`
- Extended node parameter override support in mapping layer:
  - `width`, `height`, `depth`, `fill`, `opacity`, `bandfill`, `xlabel`, `zlabel`
- Updated default model styling to be closer to PlotNeuralNet examples:
  - stronger connection style
  - softer block opacity and paper-like palette
  - better defaults for `conv/pool/block/fc/output`
- Added compact linear chain layout for long simple chains to avoid over-stretched outputs.

### Branching/Inception status
- `BranchingPlotNeuralNetRenderer` edge routing was adjusted from boxy trunks to smoother curved branch fan-out/fan-in paths.
- Current output quality is significantly cleaner than earlier rectangular routing.

### U-Net status
- `EncoderDecoderPlotNeuralNetRenderer` received multiple routing fixes:
  - cleaner handling for some `target=Up*` transitions
  - explicit filtering to preserve true encoder->concat skip links
  - skip lane behavior moved toward upper-corridor style (example-inspired)
- Note: current canonical U-Net is improved but still has local decoder-side crossing complexity in dense cases.
  - Most visible around `Concat -> Up/Dec` region.

### Regenerated outputs (representative)
- `outputs/walkthrough_resnet/walkthrough_resnet.png`
- `outputs/walkthrough_unet/walkthrough_unet.png`
- `outputs/walkthrough_inception/walkthrough_inception.png`
- `outputs/walkthrough_autoencoder/walkthrough_autoencoder.png`
- `outputs/walkthrough_gan/walkthrough_gan.png`
- `outputs/resnet_18_reference_style/resnet_18_reference_style.png`
- `outputs/resnet_34_reference_style/resnet_34_reference_style.png`
- `outputs/review_perceptron/review_perceptron.png`
- `outputs/review_unet_canonical/review_unet_canonical.png`

### Known open issues
- U-Net can still show crowded decoder-side edge geometry depending on graph shape.
- If strict parity with PlotNeuralNet `examples/Unet_Ushape` is required, next step should be:
  - enforce a rigid decoder grid contract (`Up`, `Concat`, `Dec`) per stage
  - route all skip links through stage-specific horizontal lanes with fixed drop points
  - avoid mixed interpretation of non-skip edges as skip during classification

### Immediate next recommended step
1. Implement strict stage-grid routing for canonical U-Net only.
2. Validate on:
   - `outputs/review_unet_canonical/*`
   - `outputs/walkthrough_unet/*`
3. Do not change `DualPathPlotNeuralNetRenderer` unless requested.

## 2026-04-04 - U-Net aligned toward PlotNeuralNet examples

### Context
- User asked to make U-Net look like PlotNeuralNet `examples/Unet` and `examples/Unet_Ushape`.
- Priority: preserve current renderer stack, improve routing behavior (not add a new renderer).

### Changes
- `plotneuralnet_renderer.py`
  - In `EncoderDecoderPlotNeuralNetRenderer`:
    - added `_unet_skip_lanes` map for stage-aware skip corridors.
    - restricted skip set to true `encoder -> concat` pairs in detected U-Net structure.
    - updated skip edge routing to corridor-style paths (example-inspired).
    - adjusted `target=Up*` handling to reduce false loop-like routes.
  - In `BranchingPlotNeuralNetRenderer`:
    - kept curved fan-out/fan-in routing for cleaner Inception branch geometry.

### Checks
- `python3 -m py_compile plotneuralnet_renderer.py`
- `python3 test_variety_render.py`
- Regenerated and inspected:
  - `outputs/walkthrough_unet/walkthrough_unet.png`
  - `outputs/review_unet_canonical/review_unet_canonical.png`
  - `outputs/walkthrough_inception/walkthrough_inception.png`

### Risks
- U-Net mini-graph (`walkthrough_unet`) still has visual crowding around decoder merges due to very small stage count.
- Canonical U-Net is improved, but strict parity with `examples/Unet_Ushape` requires a harder stage-grid contract for decoder geometry.

### Next
1. Implement rigid decoder grid per stage (`Up`, `Concat`, `Dec`) with fixed vertical separation.
2. Anchor skip drop points to `Concat` top anchors only (single consistent pattern).
3. Re-validate on canonical U-Net first, then back-check mini U-Net.

## 2026-04-04 - Canonical YOLO decluttering pass

### Context
- User reported YOLO output looked tangled (`canonical_yolo`): too many crossings in neck/head area.

### Changes
- `plotneuralnet_renderer.py`
  - Reworked `_build_yolo_python_script()` layout:
    - simplified stage captions by variant (`v3/v5/v8`) to shorter labels;
    - replaced dense neck fusion (`up + sum + long skip`) with cleaner `P5 -> P4 -> P3` sequence;
    - removed long `stage3/stage4 -> fuse` crossing lines;
    - removed single merged detect node; switched to per-scale outputs (`Out L/M/S`) to avoid center convergence clutter;
    - increased right-side spacing (neck/head/output offsets) to reduce local overlap.

### Checks
- `python3 -m py_compile plotneuralnet_renderer.py main.py`
- Render smoke test via renderer call:
  - `outputs/walkthrough_yolo_clean/walkthrough_yolo_clean.png`

### Notes
- Result is cleaner than previous version, but still keeps some diagonals due to 3D perspective and multi-scale branches.
- If stricter visual minimalism is needed: next pass can enforce zero-crossing topology with explicit orthogonal routing helpers for canonical YOLO.

## 2026-04-04 - YOLO style/color separation + output spacing

### Context
- User requested two targeted improvements:
  - use non-uniform colors across blocks;
  - separate output heads more.

### Changes
- `plotneuralnet_renderer.py` (`_build_yolo_python_script`)
  - Added stage-wise color redefinitions via `\\def\\ConvColor{...}`:
    - backbone: green-tinted,
    - neck: blue-tinted,
    - heads: warm orange-tinted,
    - outputs: violet-tinted.
  - Increased geometric separation on right side:
    - larger vertical spacing between `Neck P5/P4/P3`,
    - increased horizontal distance from heads to `Out L/M/S` (offset `2.5`).

### Checks
- `python3 -m py_compile plotneuralnet_renderer.py`
- Re-rendered:
  - `outputs/walkthrough_yolo_clean/walkthrough_yolo_clean.png`

## 2026-04-04 - Generic label overlap mitigation

### Context
- User reported text overlap on non-canonical generated diagrams (notably ViT linear chain).

### Changes
- `plotneuralnet_renderer.py`
  - In `_compute_linear_layout` simple-chain path:
    - replaced fixed `x_step=1.95` with adaptive spacing based on max block width and max caption line count.
  - In `_build_node_tex`:
    - multi-line captions now use `\\scriptsize\\shortstack[...]`;
    - long one-line captions (`>14` chars) are downscaled with `\\scriptsize`;
    - side input captions also switched to `\\scriptsize` and nudged farther from block.

### Validation
- `python3 -m py_compile plotneuralnet_renderer.py main.py`
- Rendered sanity sample:
  - `outputs/walkthrough_vit_textfix/walkthrough_vit_textfix.png`

### Note
- Overlap is reduced; remaining issue is readability for very long semantic labels (still dense wrapping).
