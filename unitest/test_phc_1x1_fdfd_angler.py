from core.models.layers.phc_1x1_fdfd_angler import PhC_1x1_fdfd_angler
import torch

eps_sio2 = 1.44**2
eps_si = 3.48**2

if __name__ == "__main__":
    phc = PhC_1x1_fdfd_angler(
        num_in_ports=1, 
        num_out_ports=1, 
        box_size=(10, 10), 
        port_width=(1.7320508076, 1.7320508076),  # in/out wavelength width, um
        port_len=3,  # length of in/out waveguide from PML to box. um
        border_width=1,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um. is this grid step euqal to the resolution?
        NPML=(40, 40),  # PML pixel width. pixel
        eps_r=eps_si,  # relative refractive index
        eps_bg=1.44**2,  # background refractive index
    )
    random_permittivity = torch.randn(size=(201, 201))
    phc.create_objective(1.55e-6, permittivity=random_permittivity)
    phc.create_optimzation()
    print("this is the objective function value and gradient:")
    obj = phc.obtain_objective()
    grad = phc.obtain_gradient()
    print(obj)
    print(grad)
    print(grad.shape)