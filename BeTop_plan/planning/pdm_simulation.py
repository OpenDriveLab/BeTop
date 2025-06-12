'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
import logging
import hydra
import numpy as np

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.scenario_builder import extract_scenarios_from_dataset
from nuplan.planning.script.builders.planner_builder import build_planners

from nuplan.planning.script.utils import set_default_path

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

set_default_path()

from filtering.scenario_processor import MetricProcessor
from filtering.pdm_metric import PDMMetric

from tqdm import tqdm
import csv 
import pandas as pd
    
    
class PDMEvalEngine:
    def __init__(
        self,
        planner,
        save_path,
        cache_path=None
        ):
        self.processor = MetricProcessor(with_history=20)
        self.scorer = PDMMetric()
        self.planner = planner
        self.save_path = save_path
        self.cache_path = cache_path

        self._init_stat()
    
    def _init_stat(self):

        self.detail_name = ['no_collision', 'drivable_area', 'driving_direction',
        'progress', 'ttc', 'comfortable']

        self.headers = ['log_name', 'scenario_type', 'token', 'pdm_score'] + self.detail_name

        with open(self.save_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.headers)

    def _print_stat(self):
        df = pd.read_csv(self.save_path)
        ret_dict = {
            k: np.mean(df[k].values) for k in self.headers[3:]
        }
        print(ret_dict)

    def pdm_eval_by_cache(self, scenario):

        # with open(self.cache_path + f"/{scenario.token}.pkl",'rb') as reader:
        #     metric_data = pickle.load(reader)
        metric_data = self.processor.compute_metric_cache(scenario)
        planner_input, planner_initialization = self.processor._get_planner_inputs(scenario)
        # self.planner.step_initialize(planner_initialization)
        self.planner.initialize(planner_initialization)
        plan_traj = self.planner.compute_planner_trajectory(planner_input)

        ref_score, plan_score, details = self.scorer.pdm_scoring(metric_data, plan_traj)
        
        ret_list = [
            metric_data['scenario_info']['log_name'],
            metric_data['scenario_info']['scenario_type'],
            metric_data['scenario_info']['token'],
            plan_score
        ] + [
            details[k][1] for k in self.detail_name
        ]

        with open(self.save_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(ret_list)
    
    def run(self, scenarios):

        print(f'Start PDM simulation: {len(scenarios)} scenarios')

        for scenario in tqdm(scenarios):
            self.pdm_eval_by_cache(scenario)
        
        print('Complete PDM simulation')
        self._print_stat()


@hydra.main(config_path="./config", config_name="default_simulation")
def main(cfg):

    build_logger(cfg)

    worker = build_worker(cfg)
    # fetch scenarios
    method_name = str(cfg.experiment_uid)

    scenarios = extract_scenarios_from_dataset(cfg, worker)

    planner = build_planners(cfg.planner, scenarios[0])[0]

    print('planner build!', method_name)

    sim_engine = PDMEvalEngine(
        planner,
        save_path=f'/path/to/save/pdm_eval_{method_name}.csv',
        cache_path='/pdm/cache/path'  # Adjust this path to your cache directory
    )
    sim_engine.run(scenarios)

if __name__ == "__main__":
    main()
    
