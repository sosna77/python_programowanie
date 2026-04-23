from base import *
from dataclasses import dataclass, asdict, field
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches



@dataclass
class SIRConfig(SimulationConfig):
    total_steps: int = 1000
    visualize: bool = True
    animate: int = 2                # if visualize==True -> 0 - no animation, 1 - grid animation, 2 - dashboard animation
    plots_on_screen: int = 1        # 0 - save plots, 1 - plots on screen
    state_file_name: str = 'SIR_states'
    stats_file_name: str = 'SIR_stats'
    plot_name: str = 'SIR_plot'
    animation_name: str = 'SIR_animation'
    method: int = 0                 # 0 - von Neumann, 1 - Moore's
    initial_state: int = 0          # 0 - one infected at center, 1 - random
    p_infect: float = 0.05
    p_recovery: float = 0.002
    size: int = 200


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
            if config.initial_state==1:
                p = np.random.uniform(0,1,(config.size, config.size))
                self.grid[p<config.p_infect] = 1
            else: self.grid[config.size//2, config.size//2] = 1
    
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
    colors = ["#4B4B4B", "#C14034", "#327B42"]
    my_cmap = ListedColormap(colors)

    def _SIR_animate_grid(self, config: SIRConfig, results: SIRResult, plots_dir: str):
        grids = [state.grid for state in results.steps]
        fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
        ax.axis('off')

        labels = ['S','I','R']
        legend_patches = [mpatches.Patch(color=self.colors[i], label=labels[i]) for i in range(3)]

        ax.legend(handles=legend_patches, 
                loc='upper right', 
                facecolor='black',       
                framealpha=0.5,          
                edgecolor='none',        
                labelcolor='white',      
                fontsize=10)

        im = ax.imshow(grids[0], cmap=self.my_cmap,aspect='auto', vmin=0, vmax=2)
        title = ax.text(0.05, 0.95, f'step 0', 
                transform=ax.transAxes, 
                color='white', 
                fontsize=8, 
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))
        
        def update(frame):
            im.set_data(grids[frame])
            title.set_text(f'step {frame}')

            return [im, title]

        anim = FuncAnimation(fig, update, frames=len(grids), interval=20, blit=False)
        if config.plots_on_screen: plt.show()
        else:  
            print('Writing animation to .mp4')
            mp4_path = plots_dir/f'{config.plot_name}_grid_animation.mp4'

            anim.save(mp4_path, writer='ffmpeg', fps=30, dpi=100)

            print('Animation written')
            
    def _SIR_animate_dashboard(self, config:SIRConfig, results: SIRResult, plots_dir:str):
        # data
        steps_data = [state.step for state in results.steps]
        grids = [state.grid for state in results.steps]
        s_vals = [stat.S_no for stat in results.statistics]
        i_vals = [stat.I_no for stat in results.statistics]
        r_vals = [stat.R_no for stat in results.statistics]
        new_vals = [stat.new_infections for stat in results.statistics]

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12,10))
        gs = gridspec.GridSpec(4,4)


        # grid visualization
        ax_grid = fig.add_subplot(gs[0:3, 0:3])
        ax_grid.axis('off')
        labels = ['S','I','R']
        legend_patches = [mpatches.Patch(color=self.colors[i], label=labels[i]) for i in range(3)]

        # pretty legend
        ax_grid.legend(handles=legend_patches, 
                loc='upper right', 
                facecolor='black',       
                framealpha=0.5,          
                edgecolor='none',        
                labelcolor='white',      
                fontsize=10)

        im = ax_grid.imshow(grids[0], cmap=self.my_cmap,aspect='auto', vmin=0, vmax=2)

        # updating title
        title = ax_grid.text(0.05, 0.95, f'step 0', 
                transform=ax_grid.transAxes, 
                color='white', 
                fontsize=8, 
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))
        
        # plots
        ax_s = fig.add_subplot(gs[0,3])
        line_s, = ax_s.plot([],[], label='S', color=self.colors[0])
        ax_s.legend(loc='upper right', fontsize='small')
        ax_s.set_xlim(0, config.total_steps)
        ax_s.set_ylim(0, max(s_vals))

        ax_i = fig.add_subplot(gs[1,3])
        line_i, = ax_i.plot([],[], label='I', color=self.colors[1])
        ax_i.legend(loc='upper right', fontsize='small')
        ax_i.set_xlim(0, config.total_steps)
        ax_i.set_ylim(0, max(i_vals))

        ax_r = fig.add_subplot(gs[2,3])
        line_r, = ax_r.plot([],[], label='R', color=self.colors[2])
        ax_r.legend(loc='upper right', fontsize='small')
        ax_r.set_xlim(0, config.total_steps)
        ax_r.set_ylim(0, max(r_vals))

        ax_new = fig.add_subplot(gs[3, :])
        line_new = ax_new.scatter([],[], label='new infections', color='turquoise', s=5)
        ax_new.legend(loc='upper right', fontsize='small')
        ax_new.set_xlim(0, config.total_steps)
        ax_new.set_ylim(0, max(new_vals))


        # making the plots look pretty
        for ax in [ax_s, ax_i, ax_r, ax_new]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('#777777')
            ax.spines['left'].set_color('#777777')
            
            ax.tick_params(axis='both', colors='#777777', width=0.5)

        
        def update(frame):
            im.set_data(grids[frame])
            title.set_text(f'step {frame}')

            cur_steps = steps_data[:frame+1]
            line_s.set_data(cur_steps, s_vals[:frame+1])
            line_i.set_data(cur_steps, i_vals[:frame+1])
            line_r.set_data(cur_steps, r_vals[:frame+1])
            line_new.set_offsets(np.c_[cur_steps, new_vals[:frame+1]])

            return [im, title, line_s, line_i, line_r, line_new]

        anim = FuncAnimation(fig, update, frames=len(grids), interval=20, blit=False)
        plt.tight_layout()   
        if config.plots_on_screen: plt.show()
        else:  
            print('Writing animation to .mp4')
            mp4_path = plots_dir/f'{config.plot_name}_dashboard_animation.mp4'

            anim.save(mp4_path, writer='ffmpeg', fps=30, dpi=100)

            print('Animation written')       

    def visualize(self,config: SIRConfig, results: SIRResult, plots_dir: str):
        interval = config.total_steps//25
        fig, axs = plt.subplots(5,5, figsize=(12,12), constrained_layout=True)
        for i, ax in enumerate(axs.flatten()):
            arr = results.steps[interval*i].grid
            ax.imshow(arr, cmap=self.my_cmap,aspect='auto',vmin=0, vmax=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.05, 0.95, f'step {i*interval}', 
                transform=ax.transAxes, 
                color='white', 
                fontsize=8, 
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none'))
        labels = ['S','I','R']
        legend_patches = [mpatches.Patch(color=self.colors[i], label=labels[i]) for i in range(3)]

        fig.legend(handles=legend_patches, 
                loc='upper right', 
                facecolor='black',       
                framealpha=0.5,          
                edgecolor='none',        
                labelcolor='white',      
                fontsize=10)
        fig.savefig(plots_dir/f'{config.plot_name}_grid.png',
                    dpi=300,
                    bbox_inches='tight'
                    )
        fig, axs = plt.subplots(2,2, figsize=(7,5), constrained_layout=True)

        ns = np.arange(0,len(results.steps))
        S = np.array([stat.S_no for stat in results.statistics])
        I = np.array([stat.I_no for stat in results.statistics])
        R = np.array([stat.R_no for stat in results.statistics])
        new_infections = np.array([stat.new_infections for stat in results.statistics])
        data = [S, I, R, new_infections]
        labels = ['S(t)','I(t)','R(t)','new_I(t)']
        
        for i, ax in enumerate(axs.flatten()):
            ax.plot(ns, data[i], label=labels[i], color=f'C{i}') if i!=3 else ax.scatter(ns, data[i], label=labels[i], color=f'C{i}', s=1)
            ax.set_ylabel(labels[i][:-3])
            ax.set_xlabel('step')
        if config.plots_on_screen: plt.show()
        else:  
            fig.savefig(plots_dir/f'{config.plot_name}_stat_plots.png',
                        dpi=300,
                        bbox_inches='tight'
                        )


        if config.animate==1: self._SIR_animate_grid(config=config, results=results, plots_dir=plots_dir)
        if config.animate==2: self._SIR_animate_dashboard(config=config, results=results, plots_dir=plots_dir)
        


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