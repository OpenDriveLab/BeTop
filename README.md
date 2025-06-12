# BeTop_plan

This folder contains the planning implementation of BeTop in [nuPlan](https://www.nuplan.org/nuplan).


## Highlights

Our planning pipeline contains the following features:

:white_check_mark: **Flexible Toolbox for planning tasks:**

- Training / Eval pipeline.  
- Scene Renderer: Visualize prediction/planning trajectories.

:white_check_mark: **Pipeline for popular Baselines:**

- [PlanTF](https://arxiv.org/pdf/2309.10443.pdf) reproduction.
- [PDM-Closed](https://arxiv.org/abs/2406.15349) reproduction.

## Get Started

**[NOTE]** For non-SSD storage (i.e. NAS), searching recursively across all files could be extremely slow.

Following [Issues](https://github.com/jchengai/pluto/issues/24), you may load the candidate dirs in ```.csv``` in cached metadata, and modify the ```get_local_scenario_cache``` of ```scenario_builder.py``` inside the nuplan-devkit.


You may access the pretrained weights for further tuning: [Link](https://huggingface.co/unknownuser6666/betop)

- [Installation & Data Preparation](../docs/DataPrep.md)
- [Training & Evaluation](../docs/TrainEval.md)


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