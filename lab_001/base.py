from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimulationConfig():
    total_steps: int
    dt: float 
    visualize: bool

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
    def visualize(self, results: SimulationResult):
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
        # write first state
        self.results.steps.append(self.state)
        self.results.statistics.append(self.step_analyzer.analyze_step(self.config,self.state))
        for _ in range(1, self.config.total_steps):
            next_state = self.step_rule.calculate_step(self.config, self.state)
            self.results.steps.append(next_state)
            self.state = next_state
            self.step_stats = self.step_analyzer.analyze_step(self.config,self.state)
            self.results.statistics.append(self.step_stats)

        self.final_stats = self.final_analyzer.analyze_final(self.results)
        self.results.final_statistics = self.final_stats

        if self.config.visualize: self.visalizer.visualize(self.results)
        
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