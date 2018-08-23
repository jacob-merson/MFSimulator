import yaml
import argparse
from MFSimulator.simulator import Model


def Demo(length, width, grid_density, boundary_type,
            dtype, CPU, fluid_density, fluid_velocity,
            num_bodies, body_shape, radius, major_r, minor_r,
            mass, body_centers, orientations, circulations,
            singularities, lam, rho,
            simTime, dt, save_animation, animation_format,
            animation_name):

    model = Model(length=length, width=width, grid_density=grid_density,
                boundary_type=boundary_type,
                dtype=dtype, CPU=CPU, fluid_density=fluid_density, fluid_velocity=fluid_velocity,
                num_bodies=num_bodies, body_shape=body_shape, radius=radius, major_r=major_r, minor_r=minor_r,
                mass=mass, body_centers=body_centers, orientations=orientations, circulations=circulations,
                singularities=singularities, lam=lam, rho=rho,
                simTime=simTime, dt=dt, save_animation=save_animation, animation_format=animation_format,
                animation_name=animation_name)
    
    model.save_animation = 'On'
    model.RenderAnimation('Streamlines')
    

if __name__=="__main__":
    
    parser=argparse.ArgumentParser(description = "")
    parser.add_argument('--model',help='File name of the model configuration to demo (with .yml extension)',type=argparse.FileType('r'))
    args=parser.parse_args()
    demo_params=yaml.load(args.model)

    Demo(**demo_params)








