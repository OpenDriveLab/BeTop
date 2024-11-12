# BeTopNet (WOMD Prediction)

This folder contains the full prediction implementation of BeTopNet in [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/challenges).


## Highlights

Our prediction pipeline contains the following features:

:white_check_mark: **Full support for WOMD Prediction Challenges:**

- Full pipeline for [WOMD Motion Prediction](https://waymo.com/open/challenges/2024/motion-prediction/) challenge.
- Full pipeline for [WOMD Interaction Prediction](https://waymo.com/open/challenges/2021/interaction-prediction/) challenge.

:white_check_mark: **Flexible Toolbox for prediction tasks:**

- Memory Efficient: Cached dataset and model training.
- User-Friendly Tools: Simple tools for model training/evaluation and **Leaderboard submission**.  
- Scene Renderer: Visualize predicted trajectories.

:white_check_mark: **Pipeline for reproduced popular Baselines:**

- [Wayformer](https://arxiv.org/pdf/2207.05844) baseline reproduction.
- [MTR++](https://arxiv.org/pdf/2306.17770) baseline reproduction (Deprecated).

## Get Started

- [Installation & Data Preparation](../docs//womd/DataPrep_pred.md)
- [Training & Evaluation](../docs//womd/TrainEval_pred.md)
- [Submission to Leaderboard](../docs//womd/Submission_pred.md)


## Acknowledgment / Related resources

:star_struck: We would like to thank the following repositories and works for providing a foundation for building our pipelines:

- [MTR](https://github.com/sshaoshuai/MTR)
- [UniTraj](https://github.com/vita-epfl/UniTraj)
- [TopoNet](https://github.com/OpenDriveLab/TopoNet)