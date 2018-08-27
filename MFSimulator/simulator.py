import os
import sys
import warnings
import time
import pyopencl as cl

from mako.template import Template
import argparse
import copy

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.io as sio
from scipy.spatial import distance
from scipy import linalg as alg
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

import MFSimulator.utils as tools

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

### Sets the OS-dependent matplotlib paths for saving animations to file ###

if sys.platform == 'darwin': # checks if the OS being used is a Mac
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    plt.rcParams['animation.convert_path'] = '/usr/local/bin/convert'
    
elif sys.platform == 'linux': # checks if the OS being used is Linux
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    plt.rcParams['animation.convert_path'] = '/usr/bin/convert'
    
elif sys.platform == 'win32': # checks if the OS being used is Windows
    ### THE PATH DEPENDS ON WHERE YOU INSTALLED IMAGEMAGICK AND THE VERSION INSTALLED!
    ### CHANGE THE PATH AS APPROPRIATE!
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ImageMagick-7.0.8-Q16\\ffmpeg.exe'
    plt.rcParams['animation.convert_path'] = 'C:\\Program Files\\ImageMagick-7.0.8-Q16\\convert.exe'


kernels="""

__kernel void StreamFunction(__global ${realType} *nodes_x,   // Calculates the potentials at each node
                              __global ${realType} *nodes_y,          // of the discretized domain for a given set
                              int const total_singularities,     //  of body positions.
                              float const velocity,
                              __global ${realType} *singularities_x,
                              __global ${realType} *singularities_y,
                              __global ${realType} *coefficients,
                              __global ${realType} *streamfunction)
{
    int gid =get_global_id(0);
    ${realType}2 node=(${realType}2)(nodes_x[gid],nodes_y[gid]);
    ${realType} Psi=velocity*node.y;

    for(int i=0;i<total_singularities;i++){

      ${realType}2 singularity=(${realType}2)(singularities_x[i],singularities_y[i]);
      ${realType} ai = coefficients[i];
      ${realType} r = distance(node,singularity);
      ${realType} psi = ai*log(r);
      Psi += psi;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    streamfunction[gid]=Psi;
}

__kernel void WallStreamFunction(__global ${realType} *nodes_x,   // Calculates the potentials at each node
                                  __global ${realType} *nodes_y,          // of the discretized domain for a given set
                                  int const total_singularities,     //  of body positions.
                                  ${realType} const y_center_of_mass,
                                  __global ${realType} *singularities_x,
                                  __global ${realType} *singularities_y,
                                  __global ${realType} *coefficients,
                                  __global ${realType} *streamfunction)
{
    int gid =get_global_id(0);
    ${realType}2 node=(${realType}2)(nodes_x[gid],nodes_y[gid]);
    ${realType} Psi=node.y;
    // ${realType} Psi= 0;

    // if(gid==0){
    //  printf("%f",y_center_of_mass);
    // }

    for(int i=0;i<total_singularities;i++){

      ${realType}2 singularity=(${realType}2)(singularities_x[i],singularities_y[i]);
      ${realType}2 conjugate_singularity=(${realType}2)(singularities_x[i],-1.0*singularities_y[i]-2.0*y_center_of_mass);
      ${realType} ai = coefficients[i];
      ${realType} r = distance(node,singularity);
      ${realType} conjugate_r = distance(node,conjugate_singularity);

      ${realType} psi = ai*(log(r)-log(conjugate_r));
      Psi += psi;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    streamfunction[gid]=Psi;
}

__kernel void DynamicStreamFunction(__global ${realType} *nodes_x,   // Calculates the potentials at each node
                                      __global ${realType} *nodes_y,          // of the discretized domain for a given set
                                      int const total_singularities,     //  of body positions.
                                      int const total_time__Steps,
                                      __global ${realType} *singularities_x,
                                      __global ${realType} *singularities_y,
                                      __global ${realType} *coefficients,
                                      __global ${realType} *streamfunction)
{
    int gid =get_global_id(0);
    int num_nodes = get_global_size(0);

    ${realType}2 node=(${realType}2)(nodes_x[gid],nodes_y[gid]);
    // ${realType} Psi=node.y;

    for(int t=0;t<total_time__Steps;t++){

        ${realType} Psi=node.y;

        for(int i=0;i<total_singularities;i++){

          ${realType}2 singularity=(${realType}2)(singularities_x[t*total_singularities+i],singularities_y[t*total_singularities+i]);
          ${realType} ai = coefficients[t*total_singularities+i];
          ${realType} r = distance(node,singularity);
          ${realType} psi = ai*log(r);
          Psi += psi;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        streamfunction[t*num_nodes+gid]=Psi;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void DynamicWallStreamFunction(__global ${realType} *nodes_x,   // Calculates the potentials at each node
                                      __global ${realType} *nodes_y,          // of the discretized domain for a given set
                                      int const total_singularities,     //  of body positions.
                                      ${realType} const y_center_of_mass,  // __global ${realType} *y_center_of_mass,
                                      int const total_time__Steps,
                                      __global ${realType} *singularities_x,
                                      __global ${realType} *singularities_y,
                                      __global ${realType} *coefficients,
                                      __global ${realType} *streamfunction)
{
    int gid =get_global_id(0);
    int num_nodes = get_global_size(0);

    ${realType}2 node=(${realType}2)(nodes_x[gid],nodes_y[gid]);
    // ${realType} Psi=node.y;

    for(int t=0;t<total_time__Steps;t++){

        ${realType} Psi=node.y;

        for(int i=0;i<total_singularities;i++){

          ${realType}2 singularity=(${realType}2)(singularities_x[t*total_singularities+i],singularities_y[t*total_singularities+i]);
          // ${realType}2 conjugate_singularity=(${realType}2)(singularities_x[t*total_singularities+i],-1.0*singularities_y[t*total_singularities+i]-2.0*y_center_of_mass[t]);
          ${realType}2 conjugate_singularity=(${realType}2)(singularities_x[t*total_singularities+i],-1.0*singularities_y[t*total_singularities+i]-2.0*y_center_of_mass);
          ${realType} ai = coefficients[t*total_singularities+i];
          ${realType} r = distance(node,singularity);
          ${realType} conjugate_r = distance(node,conjugate_singularity);
          ${realType} psi = ai*(log(r)-log(conjugate_r));
          Psi += psi;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        streamfunction[t*num_nodes+gid]=Psi;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

"""



class Model(object):
    """
    Implementation of the MFS numerical approximation technique to model the
    interactions and dynamics of fluid flow around cylinderical bodies.
    
    
    Methods
    -------
    Evolve()
        Method to advance the system in time from the system's initial state to
        to the specified simulation time by iteratively solving for the singularity
        coefficients, computing the body forces/moments, and then advancing the
        bodies' positions and velocities.
    PlotStreamFunction(kind = 'Flowfield')
        Computes and then plots the stream function of the flow around the initial
        (static) body positions either as just streamlines (kind = 'Streamlines')
        or as a full flow field represented by colored flow contours.
    RenderAnimation(kind='Positions')
        Method to render an animation of either just the bodies' motions
        (kind = 'Positions'), the bodies' motions and instantaneous
        streamlines (kind = 'Streamlines'), or the bodies' motions and instantaneous
        colored flow contours (kind = 'Flowfield').
    """
                
                
    def __init__(self, length=16.0, width=12.0, grid_density=10, boundary_type = 'Unbounded',
                dtype='float32', CPU = 'On', fluid_density = 1.225, fluid_velocity = 1.0,
                num_bodies = 1, body_shape = 'circle', radius = 1.5, major_r = 2.0, minor_r = 0.85,
                mass = 1.0, body_centers = [8.0,6.0], orientations = [0.0], circulations = [0.0],
                singularities = 20, lam = 2, rho = 0.9,
                simTime = 2.0, dt = 0.001, save_animation = 'Off', animation_format = 'mp4',
                animation_name = 'Stream Function Animation'):
        """
        Initializes the MFS formulation of the input system and other necessary
        attributes in preparation for either:
        
        1. The modeling of the static streamlines and/or flow contours around the
           initial flow-body system configuration
           
        2. The full dynamic simulation of the flow-induced body motions and changing
           flow contours
           
        Parameters
        ----------
        length : float
                 length (x-direction) of flow domain
        width : float
                width (y-direction) of flow domain
        grid_density : int
                       number of divisions per unit length/width for the discretized
                       flow domain mesh; only necessary if plotting streamlines and/or
                       flow fields
        boundary_type : str
                        specifies the type of (if any) domain boundary. Either
                        'Unbounded' (default) or 'Wall'
        dtype : str
                data type corresponding to the desired level of precision. Either
                'float32' for single 32-bit precision (default) or 'float64' for
                double 64-bit precision
        CPU : str
              Specifies what device (CPU or GPU) should be used for the parallelized
              stream function computation. Either 'On' (default) or 'Off'
        fluid_density : float
                        density of fluid
        fluid_velocity : float
                         speed of the fluid's uniform flow in the x-direction
        num_bodies : int
                     number of bodies in the flow model
        body_shape : str
                     specifies if the bodies are circular or elliptical cylinders.
                     Either 'circle' (default) or 'ellipse'
        radius : float
                 radius of the circular bodies (if body_shape = 'circle')
        major_r : float
                  major radius of the elliptical bodies (if body_shape = 'ellipse')
        minor_r : float
                  minor radius of the elliptical bodies (if body_shape = 'ellipse')
        mass : float
               mass of the bodies
        body_centers : numpy array
                       size (K,2) array containing the x and y coordinates of each
                       of the body's centers. K must match the num_bodies parameter
                       input
        orientations : numpy array
                       size (K,1) or (K,) array containing the orientation angles
                       (in degrees) of each of the bodies. K must match the num_bodies
                       parameter input
        circulations : numpy array
                       size (K,1) or (K,) array containing the circulations around
                       each of the bodies. K must match the num_bodies parameter input
        singularities : int
                        number of discretized singularities per body
        lam : int
              integer ratio of boundary collocation points to singularities
        rho : float
              singularity depth coefficient. Must be > 0.0 and < 1.0
        simTime : float
                  Specifies the (time) length of the simulation
        dt : float
             specifies the size of the time step in the body advancement scheme
        save_animation : str
                         specifies whether the rendered animation of the evolving
                         system should be saved to file. 'Off' (default) or 'On'
        animation_format : str
                           specifies the format of the saved animation video.
                           Either 'mp4' (default) or 'gif'
        animation_name : str
                         name of the saved animation. Can be passed with or without
                         the appropriate file extension (e.g. 'example.mp4' or 'example'
                         if animation_format = 'mp4')
        """


        self.dtype=dtype

        if dtype=='float64':  # Assures that the data type passed to the OpenCL
            self.ctype='double'     # kernel is consistent with the provided dtype
            os.environ['PYOPENCL_CTX']='0:0'
            print('64 bit CPU\n')
        else:
            self.ctype='float'
            if CPU == 'On':
                os.environ['PYOPENCL_CTX']='0:0'
                print('32 bit CPU\n')
            elif CPU == 'Off':
                os.environ['PYOPENCL_CTX']='0:1'
                print('32 bit GPU\n')
                

        self.save_animation = save_animation
        self.animation_name = animation_name
        self.animation_format = animation_format

        ### checks saved filename extension matches provided save format ###

        if animation_format == 'gif' and self.animation_name[-4:] != '.gif':
            self.animation_name = self.animation_name + '.gif'

        if animation_format == 'mp4' and self.animation_name[-4:] != '.mp4':
            self.animation_name = self.animation_name + '.mp4'


        self.length = length
        self.width = width
        self.dimensions = np.array((length,width))
        self.boundary_type = boundary_type
        self.fluid_U_velocity = fluid_velocity
        self.rho = fluid_density

        self.total_bodies = num_bodies
        self.mass = mass

        ### body properties array initialization and sizing checks ###

        self.body_centers = np.asarray(body_centers,dtype=self.dtype)
        if len(np.shape(self.body_centers)) == 1:
            self.body_centers = self.body_centers[:,np.newaxis].transpose()

        self.body_xs = self.body_centers[:,0]
        self.body_ys = self.body_centers[:,1]
        self.center_of_mass = np.mean(self.body_centers, axis=0)

        self.orientations = np.asarray(orientations,dtype=self.dtype)
        self.orientations = np.deg2rad(self.orientations)
                    
        if self.orientations.ndim == 1:
            self.orientations = self.orientations[:,np.newaxis]

        self.initial_orientations = np.copy(self.orientations)

        self.circulations = np.asarray(circulations,dtype=self.dtype)/(2.0*np.pi)
        # self.circulations = np.asarray(circulations,dtype=self.dtype)

        if any([self.body_centers.shape[0] != self.total_bodies,
                self.circulations.size != self.total_bodies,
                self.orientations.size != self.total_bodies]):
                
                if all([self.body_centers.shape[0] == self.circulations.size,
                        self.orientations.size == self.circulations.size]):
                        print("""Number of Bodies input doesn't match the sizes
                        of the other arrays. Changing value of num_bodies parameter
                        to match size of input arrays.""")
                        self.total_bodies = self.circulations.size
                        
                else:
                    
                    raise ValueError(
                    """
                    The number of body centers, circulations, and/or orientations
                    provided do not match the expected number of bodies: Check
                    num_bodies input parameter and body centers, circulations, and
                    orientations sizes.
                    """)


        self.body_vxs=np.zeros_like(self.body_xs)
        self.body_vys=np.zeros_like(self.body_ys)
        self.body_velocities = np.vstack((self.body_vxs,self.body_vys)).transpose()

        self.body_shape = body_shape
        self.singularity_depth = rho
        self.lam = lam
                
        if body_shape == 'circle':

            self.body_r=radius
            self.singularities_r=rho*radius
            self.mom_of_inertia = tools.CircularMomentOfInertia(self.mass,self.body_r)

        elif body_shape == 'ellipse':

            self.body_major_r = major_r
            self.body_minor_r = minor_r
            self.singularities_major_r = rho*major_r
            self.singularities_minor_r = rho*minor_r
            self.mom_of_inertia = tools.EllipticalMomentOfInertia(self.mass,self.body_major_r,self.body_minor_r)

        ######## converting all coordinates into complex plane ##############

        self.complex_center_of_mass = tools.Complexify(self.center_of_mass[np.newaxis,:])
        self.complex_body_centers = tools.Complexify(self.body_centers)
        self.complex_body_velocities = tools.Complexify(self.body_velocities)

        self.complex_dtype = str(self.complex_body_centers.dtype)

        ###### Flow domain grid generation (only necessary for stream line plotting) #########

        self.x_axis_points=int(grid_density*length)
        self.y_axis_points=int(grid_density*width)
        self.total_points=self.x_axis_points*self.y_axis_points

        self.x,self.y=np.linspace(0,length,self.x_axis_points,dtype=self.dtype),np.linspace(0,width,self.y_axis_points,dtype=self.dtype)
        self.uj,self.vj=np.meshgrid(self.x,self.y,indexing='xy')

        self.CoM_x = np.linspace(0-self.center_of_mass[0],length-self.center_of_mass[0],self.x_axis_points,dtype=self.dtype)
        self.CoM_y = np.linspace(0-self.center_of_mass[1],width-self.center_of_mass[1],self.y_axis_points,dtype=self.dtype)
        self.CoM_uj,self.CoM_vj=np.meshgrid(self.CoM_x,self.CoM_y,indexing='xy')

        ####################################################################

        self.dt=dt
        self.total_time = simTime
        self.t=np.arange(0,simTime,dt,dtype=self.dtype)


        self.total_singularities=int(self.total_bodies*singularities)
        self.total_collocations=lam*self.total_singularities
        self.singularities_per_body = singularities
        self.collocations_per_body = lam*singularities

        self.__GenerateComplexPoints()


        self.complex_body_centers = tools.MapToCenterOfMassFrame(self.complex_body_centers,self.complex_center_of_mass)
        self.complex_collocation_coords = tools.MapToCenterOfMassFrame(self.complex_collocation_coords,self.complex_center_of_mass)
        self.complex_singularity_coords = tools.MapToCenterOfMassFrame(self.complex_singularity_coords,self.complex_center_of_mass)

        self.all_collocation_xs = np.real(self.complex_collocation_coords)
        self.all_collocation_ys = np.imag(self.complex_collocation_coords)

        self.all_singularity_xs = np.real(self.complex_singularity_coords)
        self.all_singularity_ys = np.imag(self.complex_singularity_coords)


        self.singularity_thetas = np.angle(self.complex_singularity_coords_wrt_body_centers)
        self.collocation_thetas = np.angle(self.complex_collocation_coords_wrt_body_centers)

        ############### Dynamic Storage Arrays Creation ######################

        self.Dynamic_Singularity_Coords = np.copy(self.complex_singularity_coords)[:,np.newaxis]
        self.Dynamic_Collocation_Coords = np.copy(self.complex_collocation_coords)[:,np.newaxis]
        self.Dynamic_Coefficients = np.empty((self.total_singularities+self.total_bodies,1),dtype=self.dtype)

        self.Dynamic_Singularity_Thetas = np.copy(self.singularity_thetas)[:,np.newaxis]
        self.Dynamic_Collocation_Thetas = np.copy(self.collocation_thetas)[:,np.newaxis]

        ############################################################################

        ### PyopenCL Initialization ###

        ocl_type = {'float32': 'float',
                    'float64': 'double'}[str(dtype)]

        script = Template(kernels).render(realType=ocl_type)

        self.ctx=cl.create_some_context()
        self.queue=cl.CommandQueue(self.ctx)
        self.mf=cl.mem_flags

        self.uj_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.CoM_uj)
        self.vj_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.CoM_vj)

        self.prg=cl.Program(self.ctx,script).build()


    ################################################################ COLLOCATION AND SINGULARITY COORDINATES GENERATOR

    def __GenerateComplexPoints(self):
        """
        private class method to generate the complex coordinates of the discretized system's singularity and
        collocation points based on the input model parameters (body centers, radius, number of singularities, etc.)
        """
        
        thetas = (np.arange(self.collocations_per_body,dtype=self.dtype)*2*np.pi)/self.collocations_per_body
        thetas = thetas[:,np.newaxis]

        self.complex_collocation_coords_wrt_body_centers = np.empty([self.total_collocations,],dtype=self.complex_dtype)
        self.complex_singularity_coords_wrt_body_centers = np.empty([self.total_singularities,],dtype=self.complex_dtype)

        self.collocation_thetas_wrt_body_orientations = np.empty([self.total_collocations,1],dtype=self.dtype)
        self.singularity_thetas_wrt_body_orientations = np.empty([self.total_singularities,1],dtype=self.dtype)

        self.complex_collocation_coords = np.empty_like(self.complex_collocation_coords_wrt_body_centers)
        self.complex_singularity_coords = np.empty_like(self.complex_singularity_coords_wrt_body_centers)


        if self.body_shape == 'circle':

            for i in range(self.total_bodies):

                body_collocation_coords = self.body_r*np.exp(1j*thetas)
                body_singularity_coords = self.singularities_r*np.exp(1j*(thetas[::self.lam]+(np.pi/self.collocations_per_body)))

                self.collocation_thetas_wrt_body_orientations[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = thetas
                self.singularity_thetas_wrt_body_orientations[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = thetas[::self.lam]+(np.pi/self.collocations_per_body)

                body_collocation_coords = body_collocation_coords.flatten()
                body_singularity_coords = body_singularity_coords.flatten()

                self.complex_collocation_coords_wrt_body_centers[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = body_collocation_coords
                self.complex_singularity_coords_wrt_body_centers[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = body_singularity_coords

                self.complex_collocation_coords[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = (
                                                        self.complex_body_centers[i] + body_collocation_coords*np.exp(1j*self.orientations[i]) )

                self.complex_singularity_coords[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = (
                                                        self.complex_body_centers[i] + body_singularity_coords*np.exp(1j*self.orientations[i]) )

        if self.body_shape == 'ellipse':

            for i in range(self.total_bodies):

                body_collocation_coords = (0.5*((self.body_major_r+self.body_minor_r)*np.exp(1j*thetas)+
                                            (self.body_major_r-self.body_minor_r)*np.exp(-1j*thetas)))

                body_singularity_coords = (0.5*((self.singularities_major_r+self.singularities_minor_r)*np.exp(1j*(thetas[::self.lam]+(np.pi/self.collocations_per_body)))+
                                            (self.singularities_major_r-self.singularities_minor_r)*np.exp(-1j*(thetas[::self.lam]+(np.pi/self.collocations_per_body)))))

                self.collocation_thetas_wrt_body_orientations[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = thetas
                self.singularity_thetas_wrt_body_orientations[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = thetas[::self.lam]+(np.pi/self.collocations_per_body)

                body_collocation_coords = body_collocation_coords.flatten()
                body_singularity_coords = body_singularity_coords.flatten()

                self.complex_collocation_coords_wrt_body_centers[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = body_collocation_coords
                self.complex_singularity_coords_wrt_body_centers[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = body_singularity_coords

                self.complex_collocation_coords[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = (
                                                        self.complex_body_centers[i] + body_collocation_coords*np.exp(1j*self.orientations[i]) )

                self.complex_singularity_coords[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = (
                                                        self.complex_body_centers[i] + body_singularity_coords*np.exp(1j*self.orientations[i]) )

    ##############################################################   MATRIX ASSEMBLY AND SOLVING

    def __SolveSystem(self):
        """
        private class method that solves the discretized matrix system for the singularity
        and flux coefficients
        """

        self.__AssembleMatrix()

        M = self.LHSMatrix

        y = -1*np.append(self.fluid_U_velocity*np.imag(self.complex_collocation_coords),self.circulations)
        self.y = y

        self.coefficients, res, rnk, s = alg.lstsq(M, y)

        self.singularity_coeffs = self.coefficients[:-1*self.total_bodies]
        
        self.Dynamic_Coefficients = np.hstack((self.Dynamic_Coefficients,self.coefficients[:,np.newaxis]))
        
    def __UpdatePointCoordinates(self):
        """
        private class method to update the singularity and collocation point
        coordinates based on the current body center positions
        """

        for i in range(self.total_bodies):

            self.complex_collocation_coords[i*self.collocations_per_body:(i+1)*self.collocations_per_body] = (
                    self.complex_collocation_coords_wrt_body_centers[i*self.collocations_per_body:(i+1)*self.collocations_per_body]*
                    np.exp(1j*self.orientations[i]) + self.complex_body_centers[i])

            self.complex_singularity_coords[i*self.singularities_per_body:(i+1)*self.singularities_per_body] = (
                    self.complex_singularity_coords_wrt_body_centers[i*self.singularities_per_body:(i+1)*self.singularities_per_body]*
                    np.exp(1j*self.orientations[i]) + self.complex_body_centers[i])

        self.Dynamic_Singularity_Coords = np.hstack((self.Dynamic_Singularity_Coords,self.complex_singularity_coords[:,np.newaxis]))
        self.Dynamic_Collocation_Coords = np.hstack((self.Dynamic_Collocation_Coords,self.complex_collocation_coords[:,np.newaxis]))

        self.Dynamic_Singularity_Thetas = np.hstack((self.Dynamic_Singularity_Thetas,np.angle(self.complex_singularity_coords_wrt_body_centers)[:,np.newaxis]))
        self.Dynamic_Collocation_Thetas = np.hstack((self.Dynamic_Collocation_Thetas,np.angle(self.complex_collocation_coords_wrt_body_centers)[:,np.newaxis]))

    def __UnboundedFlowMatrix(self):
        """
        private class method that assembles the 'log |z_j - z_i|' submatrix
        in the case of an unbounded flow
        """
        
        distances = np.abs(self.complex_collocation_coords[:, None] - self.complex_singularity_coords[None, :])
        SolutionsMatrix = np.log(distances)

        return SolutionsMatrix

    def __SingleWallMatrix(self):
        """
        private class method that assembles the 'log |z_j - z_i| - log |z_j - z*_i|'
        submatrix in the case of a flow bounded by an infinite horizontal wall along
        the x-axis
        """
                
        distances = np.abs(self.complex_collocation_coords[:, None] - self.complex_singularity_coords[None, :])
        conjugate_distances = np.abs(self.complex_collocation_coords[:, None] - np.conj(self.complex_singularity_coords)[None, :]+2*self.center_of_mass[1]*1j)
        SolutionsMatrix = np.log(distances)-np.log(conjugate_distances)

        return SolutionsMatrix

    def __AssembleMatrix(self):
        """
        private class method that assembles the full LHS matrix A using serial NumPy routines
        """

        self.__UpdatePointCoordinates()

        if self.boundary_type == 'Wall' or self.boundary_type == 'wall':
            SolutionsMatrix = self.__SingleWallMatrix()
        else:
            SolutionsMatrix = self.__UnboundedFlowMatrix()

        temp = np.zeros((self.total_collocations+self.total_bodies,self.total_singularities+self.total_bodies),self.dtype)

        for i in range(self.total_bodies):

            temp[self.total_collocations+i,i*self.singularities_per_body:(i+1)*self.singularities_per_body] = 1
            temp[i*self.collocations_per_body:(i+1)*self.collocations_per_body,self.total_singularities+i] = -1

        temp[:self.total_collocations,:self.total_singularities] = SolutionsMatrix
        self.LHSMatrix = temp

    ##############################################################   STREAMLINE COMPUTATION AND PLOTTING
    
    def __DomainStreamFunction(self):
        """
        private class method that computes the stream function values at all nodes
        of the discretized flow domain given the bodies' initial (static) positions
        """

        self.__SolveSystem()

        self.coefficients_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.singularity_coeffs)
        self.all_singularity_xs = self.all_singularity_xs.astype(self.dtype)
        self.all_singularity_ys = self.all_singularity_ys.astype(self.dtype)
        self.singX_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.all_singularity_xs)
        self.singY_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.all_singularity_ys)

        self.Psi=np.empty([self.y_axis_points,self.x_axis_points],dtype=self.dtype)
        self.Psi_buffer=cl.Buffer(self.ctx,self.mf.WRITE_ONLY, self.Psi.nbytes)


        if self.boundary_type == 'Wall' or self.boundary_type == 'wall':

            self.prg.WallStreamFunction(self.queue,(self.total_points,1),None,self.uj_buffer,self.vj_buffer,np.int32(self.total_singularities),
                            self.center_of_mass[1].astype(self.dtype),self.singX_buffer,self.singY_buffer,self.coefficients_buffer,self.Psi_buffer)

        else:
            
            self.prg.StreamFunction(self.queue,(self.total_points,1),None,self.uj_buffer,self.vj_buffer,np.int32(self.total_singularities),
                            np.float32(self.fluid_U_velocity),self.singX_buffer,self.singY_buffer,self.coefficients_buffer,self.Psi_buffer)

        cl.enqueue_copy(self.queue,self.Psi,self.Psi_buffer).wait()

        if self.boundary_type == 'Wall' or self.boundary_type == 'wall':
            self.Psi = tools.MapToDomainFrame(self.Psi,self.center_of_mass[1])

    def __DynamicStreamFunction(self):
        """
        private class method that computes the stream function values at all nodes
        of the discretized flow domain at every time step of the dynamic simulation
        """

        Dynamic_singularities_reshape = self.Dynamic_Singularity_Coords[:,1::].transpose()
        Dynamic_singularity_xs = np.real(Dynamic_singularities_reshape).flatten().astype(self.dtype)
        Dynamic_singularity_ys = np.imag(Dynamic_singularities_reshape).flatten().astype(self.dtype)

        self.dynamic_singX_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=Dynamic_singularity_xs)
        self.dynamic_singY_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=Dynamic_singularity_ys)

        Dynamic_Coefficients_reshape = np.array(self.Dynamic_Coefficients[:-1*self.total_bodies,1::],copy=True).transpose().flatten()
        self.dynamic_coefficients_buffer=cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=Dynamic_Coefficients_reshape)

        self.StreamFunctions=np.empty([np.shape(self.Dynamic_Coefficients)[1],self.y_axis_points,self.x_axis_points],dtype=self.dtype)
        self.StreamFunctions_buffer=cl.Buffer(self.ctx,self.mf.WRITE_ONLY, self.StreamFunctions.nbytes)

        t_initial = time.perf_counter()

        if self.boundary_type == 'Wall' or self.boundary_type == 'wall':
            
            self.prg.DynamicWallStreamFunction(self.queue,(self.total_points,1),None,self.uj_buffer,self.vj_buffer,np.int32(self.total_singularities),
                            self.center_of_mass[1].astype(self.dtype),np.int32(np.shape(self.Dynamic_Coefficients)[1]),self.dynamic_singX_buffer,self.dynamic_singY_buffer,
                            self.dynamic_coefficients_buffer,self.StreamFunctions_buffer)
        else:
            
            self.prg.DynamicStreamFunction(self.queue,(self.total_points,1),None,self.uj_buffer,self.vj_buffer,np.int32(self.total_singularities),
                            np.int32(np.shape(self.Dynamic_Coefficients)[1]),self.dynamic_singX_buffer,self.dynamic_singY_buffer,
                            self.dynamic_coefficients_buffer,self.StreamFunctions_buffer)

        cl.enqueue_copy(self.queue,self.StreamFunctions,self.StreamFunctions_buffer).wait()

        if self.boundary_type == 'Wall' or self.boundary_type == 'wall':
            self.StreamFunctions = tools.MapToDomainFrame(self.StreamFunctions,self.center_of_mass[1])

        t_final = time.perf_counter()
        print('StreamFunction computation time:\t',t_final-t_initial)

    ##############################################################   COMPLEXIFIED COORDINATES AND FORCES/MOMENTS COMPUTATION/PLOTTING

    def __BodyForces(self,singularity_coeffs,singularity_coords,body_collocation_coords):
        """
        private class method that computes the instantaneous forces/moments acting
        on a specific body
        """
                
        body_F = np.dtype(self.complex_dtype).type(0+1j*0)
        body_M = np.dtype(self.dtype).type(0)

        for collocation_z in body_collocation_coords:

            body_F += tools.SquaredComplexPotentialDerivative(singularity_coeffs,singularity_coords,collocation_z,self.boundary_type)
            body_M += tools.SquaredComplexPotentialDerivative(singularity_coeffs,singularity_coords,collocation_z,self.boundary_type)*collocation_z

        body_F *= 0.5*1j*self.rho
        body_M = np.real(-0.5*self.rho*body_M)

        return body_F,body_M

    def __ComputeForces(self):
        """
        private class method that computes the instantaneous forces/moments acting
        on the model's bodies
        """

        for i in range(self.total_bodies):

            body_collocation_coords = self.complex_collocation_coords[i*self.collocations_per_body:(i+1)*self.collocations_per_body]
            circulation_coefficient = self.coefficients[-1*self.total_bodies+i]

            self.body_forces[i],self.body_moments[i] = self.__BodyForces(self.singularity_coeffs,self.complex_singularity_coords,body_collocation_coords)

            if circulation_coefficient > 0.0: # Direction of summation depends on circulation sign
                self.body_forces[i] *= -1
                self.body_moments[i] *= -1

        self.body_forces = np.conj(self.body_forces)
        self.body_fxs = np.real(self.body_forces)
        self.body_fys = np.imag(self.body_forces)

    ###################################################### DYNAMIC PYTHON IMPLEMENTATION!!!!

    def __InitialValues(self):
        """
        private class method that initializes the dynamic simulation's starting
        values (i.e. positions r_0,theta_0, forces F_0,M_0, and the half-step
        velocities v_1/2,omega_1/2)
        """

        self.Dynamic_complex_CoMs = np.copy(self.complex_center_of_mass)[:,np.newaxis]
        self.Complex_Positions = np.copy(self.complex_body_centers)[:,np.newaxis]
        self.Complex_Velocities = np.copy(self.complex_body_velocities)[:,np.newaxis]

        self.Dynamic_Orientations = np.copy(self.orientations)
        self.Angular_Velocities = np.zeros_like(self.Dynamic_Orientations)

        self.__SolveSystem()

        self.body_forces = np.empty((self.total_bodies,1),dtype = self.complex_dtype)
        self.body_moments = np.empty((self.total_bodies,1),dtype = self.dtype)

        self.__ComputeForces()

        self.Complex_Velocities=(self.Complex_Velocities[:,-1][:,np.newaxis]+0.5*self.dt*(1.0/self.mass)*self.body_forces)
        self.Angular_Velocities=(self.Angular_Velocities[:,-1][:,np.newaxis]+0.5*self.dt*(1.0/self.mom_of_inertia)*self.body_moments)

    def __Step(self):
        """
        private class method that advances the bodies' positions and velocities
        by one time step dt using the two-stage, staggered central-difference scheme,
        r_n+1 = r_n + dt* v_n+1/2; v_n+3/2 = v_n+1/2 + dt*(F_n+1)/m;
        theta_n+1 = theta_n + dt* omega_n+1/2; omega_n+3/2 = omega_n+1/2 + dt*(M_n+1)/I;
        """

        New_complex_positions = self.Complex_Positions[:,-1][:,np.newaxis]+self.dt*self.Complex_Velocities[:,-1][:,np.newaxis]
        New_orientations = (self.Dynamic_Orientations[:,-1]+self.dt*self.Angular_Velocities[:,-1])[:,np.newaxis]

        self.__SolveSystem()

        self.__ComputeForces()

        New_complex_velocities=(self.Complex_Velocities[:,-1][:,np.newaxis]+self.dt*(1.0/self.mass)*self.body_forces)
        New_angular_velocities=(self.Angular_Velocities[:,-1][:,np.newaxis]+self.dt*(1.0/self.mom_of_inertia)*self.body_moments)

        self.complex_body_centers,self.complex_body_velocities = New_complex_positions,New_complex_velocities
        self.orientations = New_orientations

        New_Dynamic_complex_CoMs = np.asarray(np.mean(self.complex_body_centers,axis=0))[:,np.newaxis]
        self.Dynamic_complex_CoMs = np.vstack((self.Dynamic_complex_CoMs,New_Dynamic_complex_CoMs))

        if self.body_shape == 'circle':  # checks for body collisions in the case of circular cylinders
            New_complex_positions,New_complex_velocities = self.__CheckForCollisions(New_complex_positions,New_complex_velocities)

        self.Complex_Positions=np.hstack((self.Complex_Positions,New_complex_positions))
        self.Complex_Velocities=np.hstack((self.Complex_Velocities,New_complex_velocities))

        self.Dynamic_Orientations=np.hstack((self.Dynamic_Orientations,New_orientations))
        self.Angular_Velocities=np.hstack((self.Angular_Velocities,New_angular_velocities))

    def __CheckForCollisions(self, New_complex_positions, New_complex_velocities):
        """
        private class method that, if the model's bodies are circular cylinders,
        checks if any of the bodies have collided with one another and, if so,
        exchanges their velocities and reverts their new positions to the previous
        step's positions prior to the collison (assumes a perfectly elastic collison)
        """
        
        collision_occured = False
        collisions = [];

        for i in range(self.total_bodies):
            for j in range(i,self.total_bodies):

                if i == j:
                    continue

                tol = np.finfo(np.float64).eps+2.0*self.body_r # spatial tolerance for collision to count

                if np.linalg.norm(New_complex_positions[i]-New_complex_positions[j]) < tol:
                    print('\nCOLLISION!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                    collisions.append((i,j))
                    collision_occured = True

        if collision_occured == True:
            pairs = set(tuple(sorted(l)) for l in collisions)

            for i in pairs:

                new_velocity_a = np.copy(New_complex_velocities[i[0]])

                New_complex_velocities[i[0]] = (New_complex_velocities[i[0]] -
                    (tools.ComplexInnerProduct((New_complex_velocities[i[0]]-New_complex_velocities[i[1]]),
                                                (New_complex_positions[i[0]]-New_complex_positions[i[1]]))*
                                                (New_complex_positions[i[0]]-New_complex_positions[i[1]]))/
                                                ((New_complex_positions[i[0]]-New_complex_positions[i[1]])*
                                                np.conj((New_complex_positions[i[0]]-New_complex_positions[i[1]]))))

                New_complex_velocities[i[1]] = (New_complex_velocities[i[1]] -
                    (tools.ComplexInnerProduct((New_complex_velocities[i[1]]-new_velocity_a),
                                                (New_complex_positions[i[1]]-New_complex_positions[i[0]]))*
                                                (New_complex_positions[i[1]]-New_complex_positions[i[0]]))/
                                                ((New_complex_positions[i[0]]-New_complex_positions[i[1]])*
                                                np.conj((New_complex_positions[i[0]]-New_complex_positions[i[1]]))))

            New_complex_positions = self.Complex_Positions[:,-1][:,np.newaxis]

        return New_complex_positions, New_complex_velocities

    def Evolve(self):
        """
        public class method used to run the dynamic simulation
        """

        self.__InitialValues()

        for i in range(0,len(self.t)-1):
            print(i)
            self.__Step()

        self.X_Positions,self.Y_Positions = tools.DeComplexify(self.Complex_Positions)

        self.Positions = np.empty((np.shape(self.X_Positions)[0],np.shape(self.X_Positions)[1] + np.shape(self.Y_Positions)[1]), dtype=self.X_Positions.dtype)
        self.Positions[:,0::2] = self.X_Positions
        self.Positions[:,1::2] = self.Y_Positions
        
        # deletes the first (empty) column from the final evolved system coefficients
        # array that was used to originally instantiate the storage array
        self.Dynamic_Coefficients = np.delete(self.Dynamic_Coefficients,0,1)
        
        for i in range(self.Dynamic_complex_CoMs.size):
            self.Positions[:,[2*i,2*i+1]] = tools.MapToDomainFrame(self.Positions[:,[2*i,2*i+1]],self.center_of_mass)

        self.Dynamic_Degree_Orientations = np.empty_like(self.Dynamic_Orientations)
        for i in range(self.Dynamic_Orientations.shape[1]):
            self.Dynamic_Degree_Orientations[:,i] = np.rad2deg(self.Dynamic_Orientations[:,i])

    ##############################################################   STREAMLINE COMPUTATION AND PLOTTING

    def PlotStreamFunction(self, kind = 'Flowfield'):
        """
        public class method used to plot the flow's contours and/or streamlines
        around the bodies' initial, static positions
        
        Parameters
        ----------
        kind : str
               string input to specify type of flow field figure to generate.
               'Flowfield' (default): plots both the flow's streamlines and
                                      colored contours
               'Streamlines': plots just the flow's streamlines around the bodies
        """

        if hasattr(self, 'Psi'):
            pass
        elif hasattr(self,'StreamFunctions'):
            self.Psi = self.StreamFunctions[0,:,:]
        else:
            self.__DomainStreamFunction()

        fig=plt.figure()
        
        if kind == 'Flowfield' or kind == 'flowfield':
            plt.contour(self.uj,self.vj,np.abs(self.Psi), 18,colors = 'k', zorder = 3)
            CF = plt.contourf(self.uj,self.vj,np.abs(self.Psi), 50, zorder = 2)

            plt.axis('scaled')
            ax = plt.gcf().gca()
            ax.set_xlim((0, self.dimensions[0]))
            ax.set_ylim((0, self.dimensions[1]))

            for i in range(self.total_bodies):
                if self.body_shape == 'circle':
                    body = plt.Circle((self.body_centers[i,0], self.body_centers[i,1]), self.body_r, color = 'k', zorder = 4)
                    ax.add_artist(body)

                if self.body_shape == 'ellipse':
                    body = patches.Ellipse((self.body_centers[i,0], self.body_centers[i,1]),
                                        2*self.body_major_r, 2*self.body_minor_r, np.rad2deg(self.orientations[i,0]), color = 'k', zorder=4)
                    ax.add_artist(body)

            plt.tight_layout()

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="3%")

            plt.colorbar(CF, cax=cax)
        
        elif kind == 'Streamlines' or kind == 'streamlines':
            plt.contour(self.uj,self.vj,np.abs(self.Psi), 18,colors = 'k', zorder = 3)

            plt.axis('scaled')
            ax = plt.gcf().gca()
            ax.set_xlim((0, self.dimensions[0]))
            ax.set_ylim((0, self.dimensions[1]))

            for i in range(self.total_bodies):
                if self.body_shape == 'circle':
                    body = plt.Circle((self.body_centers[i,0], self.body_centers[i,1]), self.body_r, color = 'k', zorder = 4)
                    ax.add_artist(body)

                if self.body_shape == 'ellipse':
                    body = patches.Ellipse((self.body_centers[i,0], self.body_centers[i,1]),
                                        2*self.body_major_r, 2*self.body_minor_r, np.rad2deg(self.orientations[i,0]), color = 'k', zorder=4)
                    ax.add_artist(body)

            plt.tight_layout()
            
        plt.show()

    def RenderAnimation(self,kind='Streamlines'):
        """
        public class method that renders the animation of the bodies' dynamic
        motions as they evolve in time.
        
        Parameters
        ----------
        kind : str
               string input to specify type of flow field figure to generate.
               'Positions': animates just the bodies' changing motions
                            without displaying any information about the
                            flow's contours or streamlines (fastest option)
               'Streamlines' (default): animates the bodies' motions and instantaneous
                                        streamlines (recommended)
               'Flowfield': animates the bodies' motions through the full
                            depiction of the flow field with black streamlines
                            and colored contours (longest option)
        """


        if hasattr(self, 'Positions'): # protects the simulation from being advanced
            pass                        # past the desired time length in case the
        else:                           # the system had already been previously evolved
            self.Evolve()


        if self.save_animation == 'On' or self.save_animation == 'on':
            if self.animation_format == 'mp4':
                if 'ffmpeg' in animation.writers.list() == False:
                    print('Option to save animation as .mp4 has been selected, but \
                    the necessary ''ffmpeg'' movie writer is not installed. Reverting \
                    to displaying rendered animation...')
                    self.save_animation == 'Off'
                    
            if self.animation_format == 'gif':
                if 'imagemagick' in animation.writers.list() == False:
                    print('Option to save animation as .gif has been selected, but \
                    the necessary ''imagemagick'' movie writer is not installed. Reverting \
                    to displaying rendered animation...')
                    self.save_animation == 'Off'
                    
        fig=plt.figure()

        if kind == 'Positions' or kind == 'positions':

            if self.save_animation == 'Off': # necessary to skip over a number of frames
                self.N_frames = 10           # when not saving to file due to time to render
            else:                            # next frame bottle neck on animation playback speed
                self.N_frames = 1

            ax=plt.axes(xlim=(0,self.dimensions[0]),ylim=(0,self.dimensions[1]))
            ax.set_aspect('equal')

            arrow_opts = {'mutation_scale': 20}

            def init():
                ax.scatter(self.Positions[:,0],self.Positions[:,1])
                
                for i in range(self.total_bodies):
                    
                    if self.body_shape == 'circle':
                        if self.body_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = plt.Circle((self.Positions[i,0], self.Positions[i,1]), self.body_r, color = 'k')
                        ax.add_artist(body)

                    if self.body_shape == 'ellipse':
                        if self.body_major_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = patches.Ellipse((self.Positions[i,0], self.Positions[i,1]),
                                            2*self.body_major_r, 2*self.body_minor_r, self.Dynamic_Degree_Orientations[i,0], color = 'k')
                        ax.add_artist(body)
                    
                    arrow = patches.FancyArrowPatch((self.Positions[i,0],self.Positions[i,1]),
                    (np.real(self.Dynamic_Collocation_Coords[i*self.collocations_per_body,0])+self.center_of_mass[0],
                    np.imag(self.Dynamic_Collocation_Coords[i*self.collocations_per_body,0])+self.center_of_mass[1]),
                        color='r', arrowstyle='->', linewidth=1.0, zorder = 10, **arrow_opts)

                    ax.add_artist(arrow)

            def animate(i):
                ax.cla()
                ax.collections = []
                ax.set_xlim((0,self.dimensions[0]))
                ax.set_ylim((0,self.dimensions[1]))
                ax.set_aspect('equal')
                ax.scatter(self.Positions[:,2*self.N_frames*i],self.Positions[:,2*self.N_frames*i+1])

                for j in range(self.total_bodies):

                    if self.body_shape == 'circle':
                        if self.body_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = plt.Circle((self.Positions[j,2*self.N_frames*i], self.Positions[j,2*self.N_frames*i+1]), self.body_r, color = 'k')
                        ax.add_artist(body)

                    if self.body_shape == 'ellipse':
                        if self.body_major_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = patches.Ellipse((self.Positions[j,2*self.N_frames*i], self.Positions[j,2*self.N_frames*i+1]),
                                            2*self.body_major_r, 2*self.body_minor_r, self.Dynamic_Degree_Orientations[j,self.N_frames*i], color = 'k')
                        ax.add_artist(body)

                    arrow = patches.FancyArrowPatch((self.Positions[j,2*self.N_frames*i],self.Positions[j,2*self.N_frames*i+1]),
                    (np.real(self.Dynamic_Collocation_Coords[j*self.collocations_per_body,self.N_frames*i])+self.center_of_mass[0],
                    np.imag(self.Dynamic_Collocation_Coords[j*self.collocations_per_body,self.N_frames*i])+self.center_of_mass[1]),
                        color='r', arrowstyle='->', linewidth = 1.0, zorder = 10, **arrow_opts)

                    ax.add_artist(arrow)
            
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(self.Complex_Positions.shape[1]/self.N_frames), interval = 1)
        
        if kind == 'Streamlines' or kind == 'streamlines':

            if self.save_animation == 'Off': # necessary to skip over a number of frames
                self.N_frames = 10           # when not saving to file due to time to render
            else:                            # next frame bottle neck on animation playback speed
                self.N_frames = 1

            if hasattr(self, 'StreamFunctions'):
                pass
            else:
                self.__DynamicStreamFunction()

            ax=plt.axes()
            ax.axis('scaled')

            arrow_opts = {'mutation_scale': 20}

            def init():
                ax.contour(self.uj,self.vj,np.abs(self.StreamFunctions[0,:,:]), 18,colors = 'k')

                for i in range(self.total_bodies):
                    if self.body_shape == 'circle':
                        if self.body_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = plt.Circle((self.Positions[i,0], self.Positions[i,1]), self.body_r, color = 'k')
                        ax.add_artist(body)

                    if self.body_shape == 'ellipse':
                        if self.body_major_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = patches.Ellipse((self.Positions[i,0], self.Positions[i,1]),
                                            2*self.body_major_r, 2*self.body_minor_r, self.Dynamic_Degree_Orientations[i,0], color = 'k')
                        ax.add_artist(body)

                    arrow = patches.FancyArrowPatch((self.Positions[i,0],self.Positions[i,1]),
                    (np.real(self.Dynamic_Collocation_Coords[i*self.collocations_per_body,0])+self.center_of_mass[0],
                    np.imag(self.Dynamic_Collocation_Coords[i*self.collocations_per_body,0])+self.center_of_mass[1]),
                        color='r', arrowstyle='->', linewidth=1.0, zorder = 10, **arrow_opts)

                    ax.add_artist(arrow)

            def animate(i):
                ax.cla()
                ax.collections = []
                ax.contour(self.uj,self.vj,np.abs(self.StreamFunctions[self.N_frames*i,:,:]), 18,colors = 'k')

                for j in range(self.total_bodies):
                    if self.body_shape == 'circle':
                        if self.body_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = plt.Circle((self.Positions[j,2*self.N_frames*i], self.Positions[j,2*self.N_frames*i+1]), self.body_r, color = 'k')
                        ax.add_artist(body)

                    if self.body_shape == 'ellipse':
                        if self.body_major_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = patches.Ellipse((self.Positions[j,2*self.N_frames*i], self.Positions[j,2*self.N_frames*i+1]),
                                            2*self.body_major_r, 2*self.body_minor_r, self.Dynamic_Degree_Orientations[j,self.N_frames*i], color = 'k')
                        ax.add_artist(body)

                    arrow = patches.FancyArrowPatch((self.Positions[j,2*self.N_frames*i],self.Positions[j,2*self.N_frames*i+1]),
                    (np.real(self.Dynamic_Collocation_Coords[j*self.collocations_per_body,self.N_frames*i])+self.center_of_mass[0],
                    np.imag(self.Dynamic_Collocation_Coords[j*self.collocations_per_body,self.N_frames*i])+self.center_of_mass[1]),
                        color='r', arrowstyle='->', linewidth = 1.0, zorder = 10, **arrow_opts)

                    ax.add_artist(arrow)
            
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(self.Complex_Positions.shape[1]/self.N_frames), interval = 1)

        if kind == 'Flowfield' or kind == 'flowfield':

            if self.save_animation == 'Off': # necessary to skip over a number of frames
                self.N_frames = 30           # when not saving to file due to time to render
            else:                            # next frame bottle neck on animation playback speed
                self.N_frames = 1
            
            if hasattr(self, 'StreamFunctions'):
                pass
            else:
                self.__DynamicStreamFunction()

            ax=plt.axes()
            ax.axis('scaled')

            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')

            arrow_opts = {'mutation_scale': 20}

            def init():
                ax.contour(self.uj,self.vj,np.abs(self.StreamFunctions[0,:,:]), 18,colors = 'k')
                cf = ax.contourf(self.uj,self.vj,np.abs(self.StreamFunctions[0,:,:]), 50)
                fig.colorbar(cf,cax=cax)

                for i in range(self.total_bodies):
                    if self.body_shape == 'circle':
                        if self.body_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = plt.Circle((self.Positions[i,0], self.Positions[i,1]), self.body_r, color = 'k')
                        ax.add_artist(body)

                    if self.body_shape == 'ellipse':
                        if self.body_major_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = patches.Ellipse((self.Positions[i,0], self.Positions[i,1]),
                                            2*self.body_major_r, 2*self.body_minor_r, self.Dynamic_Degree_Orientations[i,0], color = 'k')
                        ax.add_artist(body)

                    arrow = patches.FancyArrowPatch((self.Positions[i,0],self.Positions[i,1]),
                    (np.real(self.Dynamic_Collocation_Coords[i*self.collocations_per_body,0])+self.center_of_mass[0],
                    np.imag(self.Dynamic_Collocation_Coords[i*self.collocations_per_body,0])+self.center_of_mass[1]),
                        color='r', arrowstyle='->', linewidth=1.0, zorder = 10, **arrow_opts)

                    ax.add_artist(arrow)
                
            def animate(i):
                ax.cla()
                ax.collections = []
                ax.contour(self.uj,self.vj,np.abs(self.StreamFunctions[self.N_frames*i,:,:]), 18,colors = 'k')
                cf = ax.contourf(self.uj,self.vj,np.abs(self.StreamFunctions[self.N_frames*i,:,:]), 50)
                cax.cla()
                fig.colorbar(cf,cax=cax)

                for j in range(self.total_bodies):
                    if self.body_shape == 'circle':
                        if self.body_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = plt.Circle((self.Positions[j,2*self.N_frames*i], self.Positions[j,2*self.N_frames*i+1]), self.body_r, color = 'k')
                        ax.add_artist(body)

                    if self.body_shape == 'ellipse':
                        if self.body_major_r <= 1.5:
                            arrow_opts['mutation_scale']= 10
                        body = patches.Ellipse((self.Positions[j,2*self.N_frames*i], self.Positions[j,2*self.N_frames*i+1]),
                                            2*self.body_major_r, 2*self.body_minor_r, self.Dynamic_Degree_Orientations[j,self.N_frames*i], color = 'k')
                        ax.add_artist(body)

                    arrow = patches.FancyArrowPatch((self.Positions[j,2*self.N_frames*i],self.Positions[j,2*self.N_frames*i+1]),
                    (np.real(self.Dynamic_Collocation_Coords[j*self.collocations_per_body,self.N_frames*i])+self.center_of_mass[0],
                    np.imag(self.Dynamic_Collocation_Coords[j*self.collocations_per_body,self.N_frames*i])+self.center_of_mass[1]),
                        color='r', arrowstyle='->', linewidth = 1.0, zorder = 10,  **arrow_opts)

                    ax.add_artist(arrow)
                        
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(self.Complex_Positions.shape[1]/self.N_frames))
            
        if self.save_animation == 'On' or self.save_animation == 'on':
            fps = 1.0/self.dt

            if self.animation_format == 'mp4':
                anim.save(self.animation_name, fps=fps, dpi=80, writer= 'ffmpeg')
            elif self.animation_format == 'gif':
                anim.save(self.animation_name, fps=fps, dpi=80, writer= 'imagemagick')
            else:
                print('Input error: Unable to save animation. ''save_animation'' \
                parameter set to ''On'', but saved file format not specified. Please try again \
                specifying either animation_format = ''gif'' to save a gif, or animation_format = ''mp4'' \
                to save an mp4 video.')

        else:
            plt.draw()
            plt.show()




