from base import *
from dataclasses import dataclass, asdict, field
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import json


@dataclass
class SIRConfig(SimulationConfig):
    total_steps: int = 1000
    visualize: bool = False
    state_file_name: str = 'SIR_states'
    stats_file_name: str = 'SIR_stats'
    plot_name: str = 'SIR_plots.png'
    # method=0 - von Neumann
    # method=1 - Moore
    method: int = 1

    p_infect: float = 0.03
    p_recovery: float = 0.02
    size: int = 10


@dataclass
class SIRState(SimulationState):
    step: int = 0
    grid: np.ndarray = None
    new_infections: int = 0
    if_infected: np.ndarray = None
    if_recovered: np.ndarray = None

    def __post_init__(self):
        if self.grid is None:
            config = SIRConfig()

            self.grid = np.zeros(
                (config.size, config.size), 
                dtype=np.int8
                )
            self.if_infected = self.grid.copy()
            self.if_recovered = self.grid.copy()
            # for now middle infected
            self.grid[config.size//2, config.size//2] = 1
    
@dataclass
class SIRStepStatistics(StepStatistics):
    S_no: int
    I_no: int
    R_no: int
    new_infections: int

 
@dataclass
class SIRFinalStatistics(FinalStatistics):
    max_infected: int
    step_max_infected: int
    all_infected: int
    step_all_R: int

@dataclass
class SIRResult(SimulationResult):
    def write_results(self):
        stats_data = [asdict(stats) for stats in self.statistics]

        SimulationResult.setup_paths(self=self)
        state_history = np.array([state.grid for state in self.steps])
        np.save(self.state_file_path, state_history)
        with open(self.stats_file_path, 'w') as stats_file:
            json.dump(stats_data, stats_file, indent=4)

class SIRStepRule(StepRule):
    def calculate_step(self, config: SIRConfig, state: SIRState) -> SIRState:
        grid = state.grid.copy()
        S = grid==0
        I = grid==1

        ker = np.array([[0,1,0],[1,0,1],[0,1,0]]) if config.method else np.array([[1,1,1],[1,0,1],[1,1,1]])
        p = np.random.uniform(0,1,(config.size, config.size))

        infected_neighbours = (convolve2d(I.astype(int), ker, mode='same', boundary='wrap'))*S
        prob_of_infection = 1 - (1 - config.p_infect)**infected_neighbours
        newly_infected = p<prob_of_infection
        no_of_new_infections = int(np.sum(newly_infected))

        grid += newly_infected
        if_infected = state.if_infected.copy()+newly_infected
        
        p = np.random.uniform(0,1,(config.size, config.size))
        newly_recovered = p<config.p_recovery * I

        grid += newly_recovered
        if_recovered = state.if_recovered.copy()+newly_recovered

        return SIRState(step=state.step+1,
                         new_infections=no_of_new_infections,
                         grid=grid, if_infected=if_infected,
                         if_recovered=if_recovered)



class SIRStepAnalyzer(StepAnalyzer):
    def analyze_step(self, config: SIRConfig, state: SIRState) -> SIRStepStatistics:
        grid = state.grid
        S_no = int(np.sum(grid==0))
        I_no = int(np.sum(grid==1))
        R_no = int(np.sum(grid==2))
        new_infections = state.new_infections

        return (SIRStepStatistics(S_no=S_no,
                                  I_no=I_no,
                                  R_no=R_no,
                                  new_infections=new_infections))

class SIRFinalAnalyzer(FinalAnalyzer):
    def analyze_final(self, results: SimulationResult):
        infected = np.array([int((state.grid==1).sum()) for state in results.steps])
        max_infected = np.max(infected)
        steps_till_max_infected = np.argmax(infected)
        total_infected = int(np.sum(results.steps[-1].if_infected))
        all_recovered = int(np.sum(results.steps[-1].if_recovered))

        return SIRFinalStatistics(max_infected=max_infected, step_max_infected=steps_till_max_infected, 
                                  all_infected=total_infected, step_all_R=all_recovered)


class SIRVisualizer(Visualizer):
    def visualize(self,config: SIRConfig, results: SimulationResult, plots_dir: str):
        pass

def main():
    print('Beginning simulation...')
    config = SIRConfig()
    state = SIRState()
    sim = Simulation(config=config,                    
                    state=state,
                    step_rule=SIRStepRule(),
                    results=SIRResult(config=config),
                    step_analyzer=SIRStepAnalyzer(),
                    final_analyzer=SIRFinalAnalyzer(),
                    visualizer=SIRVisualizer())
    sim.run()
    print('Simmulation completed')
if __name__=='__main__':
    main()