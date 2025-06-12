# Preparation for nuPlan planning

## Installation

Install the BeTop (nuPlan) package.
Make sure the corresponding package are successfully installe

```
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt
```

Download the [nuPlan dataset](https://www.nuscenes.org/nuplan)

## Data Preparation

Following PlanTF/PLUTO, cache the dataset to accelerate training. Generate 1M frames of training data from the whole nuPlan training set. 

```
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_betop \
    scenario_builder=nuplan \
    cache.cache_path=/path/to/cache/data \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

