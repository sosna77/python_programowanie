from base import *
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import json

@dataclass
class OscillatorConfig(SimulationConfig):
    total_steps: int = 1000
    dt: float = 0.1
    visualize: bool = True
    state_file_name: str = 'oscillator_states.json'
    stats_file_name: str = 'oscillator_stats.json'
    plot_name: str = 'oscillator_plots1.png'
    # method=0 Euler
    # method=1 Verlet
    method: int = 1

    # constants
    k: float = 2
    m: float = 1
    c: float = 0.2

    # init state    
    x0: float = 17
    v0: float = 2

@dataclass
class OscillatorState(SimulationState):
    step: int = 0
    t: float = 0
    x: float = 0
    v: float = 0 
    
@dataclass
class OscillatorStateVerlet(SimulationState):
    step: int = 0
    t: float = 0
    x: float = 0
    v: float = 0 
    a: float = 0
    
@dataclass
class OscillatorStepStatistics(StepStatistics):
    E_kinetic: float 
    E_potential: float
    E_total: float

@dataclass
class OscillatorFinalStatistics(FinalStatistics):
    x_max: float
    E_end: float
    E_mean: float

@dataclass
class OscillatorResult(SimulationResult):
    def write_results(self):
        state_data = [asdict(state) for state in self.steps]
        stats_data = [asdict(stats) for stats in self.statistics]

        SimulationResult.setup_paths(self=self)

        with open(self.state_file_path, 'w') as state_file:
            json.dump(state_data, state_file, indent=4)
        with open(self.stats_file_path, 'w') as stats_file:
            json.dump(stats_data, stats_file, indent=4)

class OscillatorStepRuleEuler(StepRule):
    def calculate_step(self, config: OscillatorConfig, state: OscillatorState) -> OscillatorState:
        v = state.v + (-config.k/config.m*state.x - config.c/config.m*state.v)*config.dt
        x = state.x + v*config.dt

        return OscillatorState(step=state.step+1, t=state.t+config.dt,x=x, v=v)
    
class OscillatorStepRuleVerlet(StepRule):
    def calculate_step(self, config: OscillatorConfig, state: OscillatorStateVerlet) -> OscillatorStateVerlet:
        x = state.x + state.v * config.dt + 0.5*state.a * config.dt**2
        a = (-config.k*x - config.c*state.v)/config.m
        v = state.v + 0.5*(state.a + a) * config.dt
        return OscillatorStateVerlet(step=state.step+1, t=state.t+config.dt,x=x, v=v, a=a)

class OscillatorStepAnalyzer(StepAnalyzer):
    def analyze_step(self, config: OscillatorConfig, state: OscillatorState) -> OscillatorStepStatistics:
        E_kinetic = 0.5 * config.m * state.v**2
        E_potential = 0.5 * config.k * state.x**2
        E_total = E_kinetic + E_potential

        return (OscillatorStepStatistics(E_kinetic=E_kinetic, E_potential=E_potential, E_total=E_total))

class OscillatorFinalAnalyzer(FinalAnalyzer):
    def analyze_final(self, results: SimulationResult):
        xs = np.array([state.x for state in results.steps])
        max_x = max(xs)

        E_total = np.array([stats.E_total for stats in results.statistics])
        E_final = np.max(E_total)
        E_mean = np.mean(E_total)

        return OscillatorFinalStatistics(x_max=max_x, E_end=E_final, E_mean=E_mean)


class OscillatorVisualizer(Visualizer):
    def visualize(self,config: OscillatorConfig, results: SimulationResult, plots_dir: str):
        xs = np.array([state.x for state in results.steps])
        ts = np.array([state.t for state in results.steps])
        vs = np.array([state.v for state in results.steps])
        E = np.array([stat.E_total for stat in results.statistics])

        fig, axs = plt.subplots(2,2, figsize=(7, 5))
        ax = axs[0,0]
        ax.plot(ts, xs, color='C0')
        ax.set_title('x(t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')

        ax = axs[0,1]
        ax.plot(ts, vs, color='C1')
        ax.set_title('v(t)')
        ax.set_xlabel('t')
        ax.set_ylabel('v')
    
        ax = axs[1,0]
        ax.plot(ts, E, color='C2')
        ax.set_title('E(t)')
        ax.set_xlabel('t')
        ax.set_ylabel('E')

        ax = axs[1,1]
        ax.plot(xs, vs, color='C3')
        ax.set_title('v(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('v')              

        plt.tight_layout()
        fig.savefig(
            plots_dir/config.plot_name,
            dpi=300,
            bbox_inches='tight'
        )

def main():
    print('Beginning simulation...')
    config = OscillatorConfig()
    method = OscillatorStepRuleEuler() if config.method else OscillatorStepRuleVerlet()
    state = OscillatorState(x=config.x0, v=config.v0) if config.method else OscillatorStateVerlet(x=config.x0, v=config.v0)
    sim = Simulation(config=config,                    
                    state=state,
                    step_rule=method,
                    results=OscillatorResult(config=config),
                    step_analyzer=OscillatorStepAnalyzer(),
                    final_analyzer=OscillatorFinalAnalyzer(),
                    visualizer=OscillatorVisualizer())
    sim.run()
    print('Simmulation completed')
if __name__=='__main__':
    main()