from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path

@dataclass
class SimulationConfig():
    total_steps: int
    dt: float 
    visualize: bool
    state_file_name: str
    stats_file_name: str

@dataclass
class SimulationState(ABC):
    step: int = 0
    t: float = 0
    
    @abstractmethod
    def state_to_txt(self) -> str:
        pass

@dataclass
class StepStatistics(ABC):
    @abstractmethod
    def stats_to_txt(self) -> str:
        pass

@dataclass
class FinalStatistics():
    pass

@dataclass
class SimulationResult():
    config: SimulationConfig
    final_statistics: FinalStatistics | None = None
    steps: list[SimulationState] = field(default_factory=list)
    statistics: list[StepStatistics] = field(default_factory=list)

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
    def __init__(self, config: SimulationConfig, state: SimulationState, step_rule: StepRule,
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
        self.results = SimulationResult(config=self.config)

    def run(self):
        main_dir = Path(__file__).parent
        data_dir = main_dir/'data'
        plots_dir = main_dir/'plots'
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        state_file_path = data_dir/self.config.state_file_name
        stats_file_path = data_dir/self.config.stats_file_name

        with open(state_file_path, 'w') as states_file, open(stats_file_path, 'w') as stats_file:
            # writing headers using fields func from dataclasses
            states_file.write(",".join([field.name for field in fields(self.state)])+'\n')
            stats_file.write(",".join([field.name for field in fields(self.step_stats)])+'\n')


            # write first state
            self.results.steps.append(self.state)
            states_file.write(self.state.state_to_txt())
            self.results.statistics.append(self.step_analyzer.analyze_step(self.config,self.state))
            stats_file.write(self.step_stats.stats_to_txt())
            
            for _ in range(1, self.config.total_steps):
                next_state = self.step_rule.calculate_step(self.config, self.state)
                self.results.steps.append(next_state)
                self.state = next_state
                states_file.write(self.state.state_to_txt())
                self.step_stats = self.step_analyzer.analyze_step(self.config,self.state)
                stats_file.write(self.step_stats.stats_to_txt())
                self.results.statistics.append(self.step_stats)

            self.final_stats = self.final_analyzer.analyze_final(self.results)
            self.results.final_statistics = self.final_stats

            if self.config.visualize: self.visalizer.visualize(self.config, self.results, plots_dir)
        
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