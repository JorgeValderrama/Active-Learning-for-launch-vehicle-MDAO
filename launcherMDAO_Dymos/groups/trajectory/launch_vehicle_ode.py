import openmdao.api as om

from .components.launch_vehicle_2d_eom import LaunchVehicle2DEOM
from .components.guidance import  Guidance_pitch_over_linear, Guidance_pitch_over_exponential, Guidance_gravity_turn, Guidance_exoatmos
from .components.gravity import Gravity
from .components.thrust_losses import Thrust_losses as Thrust_losses
from .components.time_exoatmos import Time_exoatmos_a, Time_exoatmos_b
from .components.qDot import QDot

from .subgroups.orbitalParameters.orbitalParameters import OrbitalParameters
from .subgroups.aero.aero import Aero

class LaunchVehicleODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('central_body',desc = 'object of class Earth')
        self.options.declare('uncertainty_cd', types=float, desc='uncertainty about drag coefficient (-)')

    def setup(self):
        nn      = self.options['num_nodes']
        cb      = self.options['central_body']
        
        self.add_subsystem('aero', Aero(num_nodes=nn, central_body = cb,uncertainty_cd = self.options['uncertainty_cd']))
        
        self.add_subsystem('gravity', Gravity(num_nodes = nn, mu = cb.mu))
        
        self.add_subsystem('thrust_losses', Thrust_losses(num_nodes=nn))
        
        self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn, omega=cb.angularSpeed, g0 = cb.g0))
        
        self.connect('aero.Drag', 'eom.Drag')
        self.connect('gravity.g', 'eom.g')  
        
        self.connect('aero.P_a', 'thrust_losses.P_a')
        self.connect('thrust_losses.thrust','eom.thrust')
        self.connect('thrust_losses.mfr', 'eom.mfr')
        
        
class LaunchVehicleODE_lift_off(LaunchVehicleODE):

    def initialize(self):
        super().initialize()
        
        
class LaunchVehicleODE_pitch_over_linear(LaunchVehicleODE):

    def setup(self):
        
        nn      = self.options['num_nodes']
        
        self.add_subsystem('guidance', Guidance_pitch_over_linear(num_nodes=nn))
        
        super().setup()

        self.connect('guidance.theta','eom.theta')


class LaunchVehicleODE_pitch_over_exponential(LaunchVehicleODE):
        
    def setup(self):
        
        nn      = self.options['num_nodes']
        
        self.add_subsystem('guidance', Guidance_pitch_over_exponential(num_nodes=nn))
        
        super().setup()
    
        self.connect('guidance.theta','eom.theta')
        
class LaunchVehicleODE_gravity_turn(LaunchVehicleODE):
        
    def setup(self):
        
        nn      = self.options['num_nodes']

        self.add_subsystem('guidance', Guidance_gravity_turn(num_nodes=nn))
        
        super().setup()
        
        self.connect('guidance.theta','eom.theta')
        
        # ------------------------------------------
        
        self.add_subsystem('qDot', QDot(num_nodes =nn))
        
        self.connect('eom.vdot', 'qDot.vdot')
        self.connect('eom.rdot', 'qDot.rdot')
        self.connect('aero.rho', 'qDot.rho')
        self.connect('aero.d_rho_wrt_h', 'qDot.d_rho_wrt_h')
        
class LaunchVehicleODE_exoatmos_a(LaunchVehicleODE):
    
    def setup(self):
        
        nn      = self.options['num_nodes']
        
        self.add_subsystem('time_exoatmos_a', Time_exoatmos_a(num_nodes=nn))
        
        self.add_subsystem('guidance', Guidance_exoatmos(num_nodes=nn))
        
        super().setup()
        
        self.connect('time_exoatmos_a.phase_duration_total','guidance.phase_duration')
        self.connect('guidance.theta','eom.theta')
        
class LaunchVehicleODE_exoatmos_b(LaunchVehicleODE):
    
    def setup(self):
        
        nn      = self.options['num_nodes']
        cb      = self.options['central_body']
        
        self.add_subsystem('time_exoatmos_b', Time_exoatmos_b(num_nodes=nn))
        
        self.add_subsystem('guidance', Guidance_exoatmos(num_nodes=nn))
        
        self.add_subsystem('orbitalParameters', OrbitalParameters(num_nodes=nn, central_body = cb))
        
        super().setup()
        
        self.connect('time_exoatmos_b.phase_duration_total','guidance.phase_duration')
        self.connect('time_exoatmos_b.phase_time_b_shifted','guidance.phase_time')
        self.connect('guidance.theta','eom.theta')
        
# class LaunchVehicleODE_exoatmos_c(LaunchVehicleODE):
    
#     def setup(self):
        
#         nn      = self.options['num_nodes']
#         cb      = self.options['central_body']
        
#         self.add_subsystem('time_exoatmos_c', Time_exoatmos_c(num_nodes=nn))
        
#         self.add_subsystem('guidance', Guidance_exoatmos(num_nodes=nn))
        
#         self.add_subsystem('orbitalParameters', OrbitalParameters(num_nodes=nn, central_body = cb)) 
        
#         super().setup()
        
#         self.connect('time_exoatmos_c.phase_duration_total','guidance.phase_duration')
#         self.connect('time_exoatmos_c.phase_time_c_shifted','guidance.phase_time')
#         self.connect('guidance.theta','eom.theta')