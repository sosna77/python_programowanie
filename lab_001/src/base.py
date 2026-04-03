from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

@dataclass
class SimulationConfig():
    total_steps: int
    dt: float 
    visualize: bool
    state_file_name: str
    stats_file_name: str

@dataclass
class SimulationState():
    step: int = 0
    t: float = 0

@dataclass
class StepStatistics():
    pass

@dataclass
class FinalStatistics():
    pass

@dataclass
class SimulationResult(ABC):
    config: SimulationConfig
    final_statistics: FinalStatistics | None = None
    steps: list[SimulationState] = field(default_factory=list)
    statistics: list[StepStatistics] = field(default_factory=list)

    def setup_paths(self):
        main_dir = Path(__file__).parent.parent
        data_dir = main_dir/'data'
        plots_dir = main_dir/'plots'
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        self.state_file_path = data_dir/self.config.state_file_name
        self.stats_file_path = data_dir/self.config.stats_file_name

    @abstractmethod
    def write_results(self, config: SimulationConfig):
        pass

class StepRule(ABC):
    @abstractmethod
    def calculate_step(self, config: SimulationConfig, state: SimulationState):
        pass

class StepAnalyzer(ABC):
    @abstractmethod
    def analyze_step(self, config: SimulationConfig, state: SimulationState):
        pass

class FinalAnalyzer(ABC):
    @abstractmethod
    def analyze_final(self, results: SimulationResult):
        pass

class Visualizer(ABC):
    @abstractmethod
    def visualize(self,config: SimulationConfig, results: SimulationResult, plots_dir: str):
        pass

class Simulation:
    def __init__(self, config: SimulationConfig, state: SimulationState, step_rule: StepRule, results: SimulationResult,
                 step_analyzer: StepAnalyzer, final_analyzer: FinalAnalyzer, visualizer: Visualizer):
        
        # initial state
        self.config = config
        self.state = state

        # classes for actions
        self.step_rule = step_rule
        self.step_analyzer = step_analyzer
        self.final_analyzer = final_analyzer
        self.visalizer = visualizer

        # create containers
        self.step_stats = self.step_analyzer.analyze_step(self.config, self.state)
        self.results = results

    def run(self):
        main_dir = Path(__file__).parent.parent
        data_dir = main_dir/'data'
        plots_dir = main_dir/'plots'
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        state_file_path = data_dir/self.config.state_file_name
        stats_file_path = data_dir/self.config.stats_file_name

        # write first state
        self.results.steps.append(self.state)
        self.results.statistics.append(self.step_analyzer.analyze_step(self.config,self.state))
        
        for _ in range(1, self.config.total_steps):
            next_state = self.step_rule.calculate_step(self.config, self.state)
            self.results.steps.append(next_state)
            self.state = next_state
            self.step_stats = self.step_analyzer.analyze_step(self.config,self.state)
            self.results.statistics.append(self.step_stats)


        print('Writing results...')
        self.results.write_results(self.config)
        print('Writing final statistics...')
        self.final_stats = self.final_analyzer.analyze_final(self.results)
        self.results.final_statistics = self.final_stats
        print(asdict(self.final_stats))

        if self.config.visualize: 
            self.visalizer.visualize(self.config, self.results, plots_dir)
            print('Saving visualizations...')
        
__all__ = [
    "SimulationConfig", 
    "SimulationState", 
    "StepStatistics", 
    "FinalStatistics", 
    "SimulationResult", 
    "StepRule", 
    "StepAnalyzer", 
    "FinalAnalyzer", 
    "Visualizer",
    "Simulation"
]