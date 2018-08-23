import numpy as np


def Complexify(coordinates_array):
    """
    function to convert an (N,2) real array to an (N,1) complex array
    
    Parameters
    ---------
    coordinates_array : ndarray
                        2D (N,2) array
                        
    Returns
    -------
    complex_array : ndarray
                    1D (N,1) complex array
    """
    complex_array = coordinates_array[:,0]+1j*coordinates_array[:,1]
    return complex_array
    
def DeComplexify(complex_array):
    """
    function to convert an (N,1) complex array to an (N,2) real array
    
    Parameters
    ---------
    complex_array : ndarray
                    1D (N,1) complex array
                        
    Returns
    -------
    real_coordinate_array : ndarray
                            1D (N,1) array containing the real
                            components of complex_array
    imaginary_coordinate_array : ndarray
                                 1D (N,1) array containing the imaginary
                                 components of complex_array
    """
    real_coordinate_array = np.real(complex_array)
    imaginary_coordinate_array = np.imag(complex_array)
    return real_coordinate_array, imaginary_coordinate_array

def SquaredComplexPotentialDerivative(singularity_coeffs,singularity_coords,z,boundary_type):
    """
    function to return the square of the complex potential derivative with respect
    to complex coordinate z
    
    Parameters
    ---------
    singularity_coeffs : ndarray
                         a size (N,1) or (N,) array containing the singularity
                         coefficients a_i
    singularity_coords : ndarray
                         a size (N,1) or (N,) complex array containing the
                         complex coordinates of the singularities
    z : cfloat
        complex floating point coordinate
    boundary_type : str
                    string specifying the model's boundary type
                    
    Returns
    -------
    dw_dz_squared : cfloat
                    square of the complex potential derivative w(z)
    """
    if boundary_type == 'Unbounded':
        dw_dz = UnboundedFlow(singularity_coeffs,singularity_coords,z)
    elif boundary_type == 'Wall':
        dw_dz = SingleWallBoundary(singularity_coeffs,singularity_coords,z)
    
    dw_dz_squared = dw_dz*dw_dz
    return dw_dz_squared
    
def UnboundedFlow(singularity_coeffs,singularity_coords,z):
    """
    function to return the complex potential derivative with respect
    to complex coordinate z for an unbounded flow
    
    Parameters
    ---------
    singularity_coeffs : ndarray
                         a size (N,1) or (N,) array containing the singularity
                         coefficients a_i
    singularity_coords : ndarray
                         a size (N,1) or (N,) complex array containing the
                         complex coordinates of the singularities
    z : cfloat
        complex floating point coordinate
        
    Returns
    -------
    dw_dz : cfloat
            complex potential derivative w(z)
    """
    denom = z-singularity_coords
    dw_dz = np.dot(singularity_coeffs,1/denom)
    return dw_dz
    
def SingleWallBoundary(singularity_coeffs,singularity_coords,z):
    """
    function to return the complex potential derivative with respect
    to complex coordinate z for a flow bounded by a horizontal wall along
    the x-axis
    
    Parameters
    ---------
    singularity_coeffs : ndarray
                         a size (N,1) or (N,) array containing the singularity
                         coefficients a_i
    singularity_coords : ndarray
                         a size (N,1) or (N,) complex array containing the
                         complex coordinates of the singularities
    z : cfloat
        complex floating point coordinate
        
    Returns
    -------
    dw_dz : cfloat
            complex potential derivative w(z)
    """
    dw_dz_plus = UnboundedFlow(singularity_coeffs,singularity_coords,z)
    dw_dz_minus = UnboundedFlow(singularity_coeffs,np.conj(singularity_coords),z)
    dw_dz = dw_dz_plus - dw_dz_minus
    return dw_dz
        
def MapToCenterOfMassFrame(point_coordinates,CoM_coordinates):
    """
    function to map a set of point coordinates from the global 'lab'
    reference frame to the inertial center-of-mass reference frame
    
    Parameters
    ---------
    point_coordinates : ndarray
                        2D (N,2) array containing global coordinates
    CoM_coordinates : ndarray
                      2D (1,2) or (2,) array containing coordinates of
                      the center of mass of the system
    
    Returns
    -------
    CoM_point_coordinates : ndarray
                            2D (N,2) array containing mapped coordinates
    """
    CoM_point_coordinates = point_coordinates - CoM_coordinates
    return CoM_point_coordinates
    
def MapToDomainFrame(CoM_point_coordinates,CoM_coordinates):
    """
    function to map a set of point coordinates from the inertial
    center-of-mass reference frame to the global 'lab' reference frame
    
    Parameters
    ---------
    CoM_point_coordinates : ndarray
                            2D (N,2) array containing coordinates with
                            respect to the inertial reference frame
    CoM_coordinates : ndarray
                      2D (1,2) or (2,) array containing coordinates of
                      the center of mass of the system
    
    Returns
    -------
    domain_point_coordinates : ndarray
                            2D (N,2) array containing mapped coordinates
    """
    domain_point_coordinates = CoM_point_coordinates + CoM_coordinates
    return domain_point_coordinates
    
def ComplexInnerProduct(complex1,complex2):
    """
    function to compute the complex inner product of two complex numbers/arrays
    of complex numbers
    
    Parameters
    ----------
        complex1 : ndarray
                   size (N,) or (N,1) complex array
        complex2 : ndarray
                   size (N,) or (N,1) complex array
                   
    Returns
    -------
        complex_inner_product : cfloat
                                complex inner product of arrays complex1 and complex2
    """
    complex_inner_product = np.real(complex1)*np.real(complex2)+np.imag(complex1)*np.imag(complex2)
    return complex_inner_product

def CircularMomentOfInertia(mass,radius):
    """
    function to compute the corresponding moment of inertia
    of a circular cylinder according to the formula I = 1/2 mr^2
    
    Parameters
    ----------
        mass : float
               mass of the circular cylinder
        radius : float
                 radius of the circular cylinder
                 
    Returns
    -------
        mom_of_inertia : float
                         moment of inertia
    """
    mom_of_inertia = 0.5*mass*radius*radius
    return mom_of_inertia
    
def EllipticalMomentOfInertia(mass,major_radius,minor_radius):
    """
    function to compute the corresponding moment of inertia
    of an elliptical cylinder according to the
    formula I = 1/4m(r_major^2 + r_minor^2)
    
    Parameters
    ----------
        mass : float
               mass of the elliptical cylinder
        major_radius : float
                       major radius of the elliptical cylinder
        minor_radius : float
                       minor radius of the elliptical cylinder
                 
    Returns
    -------
        mom_of_inertia : float
                         moment of inertia
    """
    mom_of_inertia = 0.25*mass*(major_radius*major_radius + minor_radius*minor_radius)
    return mom_of_inertia




