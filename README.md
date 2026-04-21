# brain_uav_paper2

Independent experiment scaffold for Paper 2 of the brain-inspired UAV project.

## 1. Project Overview

This repository is an independent experimental codebase for Paper 2.  
Its goal is to support a brain-inspired visual perception pipeline for target-related online trajectory planning under the same aircraft and no-fly-zone modeling framework used in Paper 1.

At the current stage, this repository focuses on:

- freezing interfaces and experiment protocol;
- preparing a synthetic visual data generation pipeline;
- supporting public dataset based external visual validation;
- reserving a bridge for future closed-loop integration after Paper 1 physical calibration is finalized.

## 2. Current Status

Current progress:

- independent Python project scaffold has been created;
- package structure has been initialized;
- interface layer has been frozen;
- experiment protocol layer has been frozen;
- unified schema has been defined;
- evaluation metric names have been defined;
- Git version control and GitHub remote sync are ready.

What is **not finalized yet**:

- final physical world unit;
- final time step `dt`;
- final aircraft speed;
- final no-fly-zone radius scale;
- final render GSD;
- final closed-loop metric formulas;
- final bridge implementation to the corrected Paper 1 environment.

## 3. Repository Structure

```text
brain_uav_paper2/
├─ configs/                  # configuration files
│  ├─ env.yaml               # environment config schema placeholder
│  ├─ render.yaml            # rendering config schema placeholder
│  ├─ vision.yaml            # visual task config schema placeholder
│  └─ experiment.yaml        # experiment protocol freeze
├─ docs/
│  └─ phase0_freeze.md       # phase 0 freeze note
├─ scripts/
│  ├─ check_interface.py     # phase 0A interface check
│  └─ check_phase0b.py       # phase 0B protocol check
├─ src/paper2/
│  ├─ common/
│  │  ├─ config.py           # yaml loader
│  │  └─ types.py            # core state definitions
│  ├─ datasets/
│  │  └─ unified_schema.py   # unified record schema
│  ├─ env_adapter/
│  │  ├─ interfaces.py       # environment protocol
│  │  └─ paper1_bridge.py    # placeholder bridge to paper1 environment
│  ├─ eval/
│  │  └─ metrics_spec.py     # metric name definitions
│  ├─ render/                # rendering module (to be implemented)
│  ├─ tracking/              # tracking module (to be implemented)
│  └─ vision/                # vision module (to be implemented)
├─ outputs/                  # generated outputs
├─ tests/                    # tests
├─ .gitignore
└─ pyproject.toml