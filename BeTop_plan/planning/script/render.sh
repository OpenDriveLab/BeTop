
PLANNER="BeTop"

python renderer.py \
    +simulation=open_loop_boxes \
    scenario_builder=nuplan_challenge \
    scenario_filter=test14-hard \
    worker.threads_per_node=32 \
    planner=$PLANNER \
    verbose=true \
    planner.imitation_planner.planner_ckpt='/path/to/checkpoint/last.ckpt'