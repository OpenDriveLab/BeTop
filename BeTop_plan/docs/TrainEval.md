# Training & Evaluation for nuPlan planning

## Training

Run the following command to train the planner:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/path/to/data/root"
export NUPLAN_MAPS_ROOT="/path/to/data/root/maps"
export NUPLAN_EXP_ROOT="/path/to/experiment/log/root"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /planning/run_training.py\
  py_func=train +training=train_betop \
  worker=single_machine_thread_pool worker.max_workers=128 \
  scenario_builder=nuplan cache.cache_path=/path/to/cache/data \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32  data_loader.params.num_workers=32  \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=disabled
```

## Eval

Modify and run the eval scripts in ```/planning/script```

## Visualization/Rendering

run the ```render.sh``` to save customed per-frame rendering or nuboard for whole scene visualizations.