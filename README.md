<h1 align="center"> BeTop: Reasoning Multi-Agent Behavioral Topology 
for Interactive Autonomous Driving </h1> 

<div align="center">
 
[![arXiv](https://img.shields.io/badge/arXiv-2409.18031-479ee2.svg)](https://arxiv.org/abs/2409.18031)

</div>

<div id="top" align="center">
<p align="center">
<img src="assets/betop_teaser.png" width="1000px" >
</p>
</div>

> - [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=zh-CN), Yu Qiao, Chen Lv and Hongyang Li  
> - [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a862f5788fd09bb6843c694d8120d50c-Abstract-Conference.html) | [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/95974.png?t=1731577555.2413552) | [Challenge Report](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20Interaction%20Prediction%20Challenge%20-%20Honorable%20Mention%20-%20BeTop-ens.pdf)  
> - If you have any questions, please feel free to contact: *Haochen Liu* ( haochen002@e.ntu.edu.sg )

<!-- > 📜 Preprint: <a href="https://arxiv.org/abs/2409.09016"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a> -->

---
**[2025-06]** The ensembled version of BeTop ```BeTop-ens``` has received 3rd place of [2025 WOMD Interaction Prediction Challenge](https://waymo.com/open/challenges/). [Report](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20Interaction%20Prediction%20Challenge%20-%20Honorable%20Mention%20-%20BeTop-ens.pdf)

**[2024-11]** Scenario Token released for ```Test14-Inter```. [Link](https://github.com/OpenDriveLab/BeTop/releases/download/nuplan/test14-inter.yaml)

**[2024-11]** Prediction project released.

<!-- Full code and checkpoints release is coming soon. Please stay tuned. -->

## Overview

**BeTop**  leverages braid theory to model multi-agent future behaviors in autonomous driving;

<div id="top" align="center">
<p align="center">
<img src="assets/betop.png" width="1000px" >
</p>
</div>

The synergetic framework, *BeTopNet*, integrates topology reasoning with prediction and planning tasks for autonomous driving.

<div id="top" align="center">
<p align="center">
<img src="assets/betopnet.png" width="1000px" >
</p>
</div>

## Get Started

### Prediction

We provide the full prediction implementation of BeTopNet in Waymo Open Motion Dataset (WOMD).

**Features:**
- :white_check_mark: Full support for WOMD Prediction Challenges  
- :white_check_mark: Flexible Toolbox for prediction tasks  
- :white_check_mark: Pipeline for reproduced popular Baselines

**[Guideline](womd/README.md)**
   - [Data Preparation](/docs/womd/DataPrep_pred.md)
   - [Training & Evaluation](/docs//womd/TrainEval_pred.md)
   - [Testing & Submission](/docs//womd/Submission_pred.md)

## TODO List

- [x] Initial release
- [x] Prediction pipeline in WOMD
- [x] Planning pipeline in nuPlan

## Citation

If you find the project helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{liu2024betop,
 title={Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving}, 
 author={Haochen Liu and Li Chen and Yu Qiao and Chen Lv and Hongyang Li},
 booktitle={NeurIPS},
 year={2024}
}
```
