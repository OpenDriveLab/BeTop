cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER="BeTop"
SPLIT='test14-hard'
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents" 
for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=24 \
        experiment_uid=$SPLIT/$planner \
        worker=ray_distributed \
        worker.threads_per_node=128 \
        distributed_mode='SINGLE_NODE' \
        number_of_gpus_allocated_per_simulation=0.15 \
        enable_simulation_progress_bar=true \
        verbose=true \
        planner.simulation_metric=$challenge \
        planner.imitation_planner.planner_ckpt='/path/to/checkpoints/xxx.ckpt'
done