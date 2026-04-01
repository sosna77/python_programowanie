from base import *
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class OscilatorConfig(SimulationConfig):
    total_steps: int = 1
    dt: float = 0.1

    # constants
    k: float = 1
    m: float = 1
    c: float = 1

    # init state    
    x0: float = 0
    v0: float = 0

    visualize: bool = False

@dataclass
class OscilatorState(SimulationState):
    step: int = 0
    t: float = 0
    x: float = 0
    v: float = 0 

    def state_to_txt(self):
        return f'{self.step},{self.t},{self.x},{self.v}\n'
    
@dataclass
class OscilatorStepStatistics(StepStatistics):
    E_kinetic: float 
    E_potential: float
    E_total: float

    def stats_to_txt(self):
        return f'{self.E_kinetic},{self.E_potential},{self.E_total}'

@dataclass
class OscilatorFinalStatistics(FinalStatistics):
    x_max: float
    E_end: float
    E_mean: float

class OscilatorStepRule(StepRule):
    def calculate_step(self, config: OscilatorConfig, state: OscilatorState) -> OscilatorState:
        v = state.v + (-config.k/config.m*state.x - config.c/config.m*state.v)*config.dt
        x = state.x + v*config.dt

        return OscilatorState(step=state.step+1, t=state.t+config.dt,x=x, v=v)

class OscilatorStepAnalyzer(StepAnalyzer):
    def analyze_step(self, config: OscilatorConfig, state: OscilatorState) -> OscilatorStepStatistics:
        E_kinetic = 0.5 * config.m * state.v**2
        E_potential = 0.5 * config.k * state.x**2
        E_total = E_kinetic + E_potential

        return (OscilatorStepStatistics(E_kinetic=E_kinetic, E_potential=E_potential, E_total=E_total))

class OscilatorFinalAnalyzer(FinalAnalyzer):
    def analyze_final(self, results: SimulationResult):
        xs = np.array([state.x for state in results.steps])
        max_x = max(xs)

        E_total = np.array([stats.E_total for stats in results.statistics])
        E_final = np.max(E_total)
        E_mean = np.mean(E_total)

        return OscilatorFinalStatistics(x_max=max_x, E_end=E_final, E_mean=E_mean)


class OscilatorVisualizer(Visualizer):
    def visualize(self, results: SimulationResult):
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
        plt.show()

def main():
    print('Beginning simulation...')
    config = OscilatorConfig(total_steps=1000, dt=0.1, k=0.5, m=1, c=0.3, x0=7, v0=0, visualize=True)
    sim = Simulation(config=config,                    
                    state=OscilatorState(x=config.x0, v=config.v0),
                    step_rule=OscilatorStepRule(),
                    step_analyzer=OscilatorStepAnalyzer(),
                    final_analyzer=OscilatorFinalAnalyzer(),
                    visualizer=OscilatorVisualizer())
    sim.run()
    print(sim.results.final_statistics)
    print('Simmulation completed')
if __name__=='__main__':
    main()