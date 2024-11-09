from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from astropy import units
import galpy
from galpy.util import coords
from galpy.orbit import Orbit
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord 

class ForwardIntegration:
    """
    Class for forward and backward integrations of satellite orbits.
    """
    def __init__(self, cart_pos, tan_theta, tf, pot, N_realization = 1):
      
        """
        Initialize the Integration class with initial Cartesian positions and other parameters.

        :param cart_pos: numpy array of Cartesian coordinates (x, y, z) for each element.
        :param tan_theta: Angular range for velocity distribution.
        :param vel_pot: Gravitational potential used for velocity calculations.
        """
        self.pot = pot
        self.tf = tf
        self.N_realization  = N_realization
        
        self.num, _ = cart_pos.shape
        self.x = cart_pos[:,0]
        self.y = cart_pos[:,1]
        self.z = cart_pos[:,2]
        
        self.tan_theta = tan_theta
        
        self.v_x = None
        self.v_y = None
        self.v_z = None
        
        self.vx_array = None
        self.vz_array = None
        self.vy_array = None
        
        self.xfor_evo = None
        self.yfor_evo = None
        self.zfor_evo = None
        
        self.ra_last = None
        self.dec_last = None
        self.dist_last =  None
        self.pmra_last = None
        self.pmdec_last =  None
        self.vlos_last = None
       
        self.o_for_array = None
        
        self.for_magA =None
        self.for_magB =None
        self.for_magC =None
        self.ca_for_mean = None
        self.ca_for_std = None
        self.ca_for_all = None
        
        self.unbound_sat_for_mean = None
        self.unbound_sat_for_std = None
        self._get_forward_integration()               

       
    def _uniform_range(self, range_distr, range_excluded, N):
        """
        Generate a random sample of size N from a uniform distribution over a specified range,
        excluding a specific sub-range.

        :param range_distr: The overall range as a tuple (min, max).
        :param range_excluded: The sub-range to exclude as a tuple (min, max).
        :param N: The number of random samples to generate.
        :return: A numpy array of random samples.
        """
        # Split the range into two parts, excluding the specified range
        lower_range = (range_distr[0], range_excluded[0])
        upper_range = (range_excluded[1], range_distr[1])

        # Calculate the length of each range
        lower_range_length = lower_range[1] - lower_range[0]
        upper_range_length = upper_range[1] - upper_range[0]
        total_length = lower_range_length + upper_range_length

        # Calculate the number of samples to draw from each part
        N_lower = int(N * (lower_range_length / total_length))
        N_upper = N - N_lower

        # Generate samples
        lower_samples = np.random.uniform(lower_range[0], lower_range[1], N_lower)
        upper_samples = np.random.uniform(upper_range[0], upper_range[1], N_upper)

        # Merge the samples
        samples = np.concatenate([lower_samples, upper_samples])
        np.random.shuffle(samples)  # Shuffle to ensure randomness

        return samples


    def _rotate_vector(self, vec1, vec2, theta):
        """
        Rotate a vector around an axis by a specified angle.

        :param vec1: The vector to be rotated (3-element array).
        :param vec2: The axis vector around which rotation occurs (3-element array).
        :param theta: Rotation angle in radians.
        :return: Rotated vector as a numpy array.
        """

        # Check if both vectors have three elements
        if len(vec1) != 3 or len(vec2) != 3:
            raise ValueError("Both vec1 and vec2 must have exactly three elements.")

        # Normalize vec2
        norm = np.linalg.norm(vec2)
        vec2 = vec2 / norm if norm > 0 else vec2

        # Rodrigues' rotation formula
        rotated_vec1 = vec1 * np.cos(theta) + \
                    np.cross(vec2, vec1) * np.sin(theta) + \
                    vec2 * np.dot(vec2, vec1) * (1 - np.cos(theta))

        return rotated_vec1

    
    
    def _get_unit_vectors(self):
        """
        Calculate radial, tangential, and perpendicular unit vectors for each position.

        This method computes the unit vectors based on the current satellite positions. The radial unit vectors
        are calculated relative to the origin. It uses the tensor-of-inertia to determine the normal vector,
        which in turn is used to calculate tangential and perpendicular unit vectors.

        Returns:
            tuple: A tuple of three numpy arrays representing the radial, tangential, and perpendicular
                   unit vectors for each satellite position.

        Raises:
            ValueError: If Cartesian coordinates are not set before calling this method.
        """
        if self.x is None or self.y is None or self.z is None:
            raise ValueError("Cartesian coordinates not set. Call `get_cartesian_coordinates()` first.")
        
        pos  = np.vstack((self.x, self.y, self.z)).T

        # Obtain the normal vector using the tensor of inertia function
        normal_vec = toi(pos)["AxisC"] 

        # Calculate the magnitude (norm) of each position vector
        mag_r = np.linalg.norm(pos, axis=1)
        # Normalize the position vectors to get radial unit vectors
        radial_unit = pos / mag_r[:, np.newaxis]

        # Compute the cross product of radial unit vectors with the normal vector to get tangential vectors
        tang_v = np.cross(radial_unit, normal_vec)
        # Normalize the tangential vectors
        mag_tan = np.linalg.norm(tang_v, axis=1)
        tan_unit = tang_v / mag_tan[:, np.newaxis]

        # Compute the cross product of tangential and radial unit vectors to get perpendicular vectors
        perp_v = np.cross(tan_unit, radial_unit)
        # Normalize the perpendicular vectors
        mag_perp = np.linalg.norm(perp_v, axis=1)
        perp_unit = perp_v / mag_perp[:, np.newaxis]

        return radial_unit, tan_unit, perp_unit
    
    def _uniform(self, theta, N):
        """
        Generate random values from a uniform distribution.

        :param theta: Angular range for velocity distribution.
        :param N: Number of random samples.
        :return: Numpy array of random values.
        """        
        return np.random.uniform(-theta, theta, N)
    
    
    def _velocities(self, theta, pot):
        """
        Calculate the velocity vectors for each position.

        This method computes the velocities based on the satellite positions, a random theta angle, and the
        provided potential. It calculates radial, tangential, and perpendicular velocities and combines them
        to obtain the Cartesian velocity components.

        :param theta: Angular range for velocity distribution.
        :param pot: Gravitational potential used for velocity calculations.
        :return: Cartesian components of velocity as numpy arrays (v_x, v_y, v_z).
        """
        rad_unit, old_tan_unit, perp_unit = self._get_unit_vectors()

        if theta == 0:
            theta_tan = np.radians(self._uniform(theta, self.num ))
        else:
            theta_tan = np.radians(self._uniform_range([-theta, theta], [-20, 20], self.num))

        tan_unit = np.array([self._rotate_vector(old_tan_unit[i, :], perp_unit[i, :], 
                                           theta_tan[i]) for i in range(self.num)])

    
        pos = np.vstack((self.x*units.kpc, self.y*units.kpc, self.z*units.kpc)).T
        radial = np.linalg.norm(pos, axis=1)
        vcir = galpy.potential.vcirc(pot, radial) * 220. * units.km / units.s  # Magnitude of Circular velocity
        rad_mag = np.zeros(self.num)  # No velocity in Radial direction
        tan_mag = vcir.value  # Velocity in Tangential direction
        per_mag = np.zeros(self.num)  # No velocity in Perpendicular direction

        rad_vel = rad_unit * rad_mag[:, np.newaxis]
        per_vel = perp_unit * per_mag[:, np.newaxis]
        tan_vel = tan_unit * tan_mag[:, np.newaxis]

        # Cartesian Components
        v_x = (tan_vel[:, 0] + rad_vel[:, 0] + per_vel[:, 0]) 
        v_y = (tan_vel[:, 1] + rad_vel[:, 1] + per_vel[:, 1]) 
        v_z = (tan_vel[:, 2] + rad_vel[:, 2] + per_vel[:, 2])
        return v_x, v_y, v_z
    
    def _cart_evol(self, tf):
        """
        Calculate the evolution of Cartesian coordinates for the forward integration.

        :param tf: Array of time values for integration.
        """
        self.xfor_evo = np.empty((self.N_realization, self.num, len(tf)))
        self.yfor_evo = np.empty((self.N_realization, self.num, len(tf)))
        self.zfor_evo = np.empty((self.N_realization, self.num, len(tf)))
        
        for n in range(self.N_realization):
            self.xfor_evo[n, :] =self.o_for_array[n].x(tf)
            self.yfor_evo[n, :] =self.o_for_array[n].y(tf)
            self.zfor_evo[n, :] =self.o_for_array[n].z(tf)
            
    
    def _forward_last(self, tf):
        """
        Calculate the last values for the forward integration.

        :param tf: Array of time values for integration.
        """
        
        self.ra_last =  np.empty((self.N_realization, self.num ))
        self.dec_last =  np.empty((self.N_realization, self.num))
        self.dist_last =  np.empty((self.N_realization, self.num))
        self.pmra_last =   np.empty((self.N_realization, self.num))
        self.pmdec_last =  np.empty((self.N_realization, self.num))
        self.vlos_last =  np.empty((self.N_realization, self.num))
        
        
        for n in range(self.N_realization):
            self.ra_last[n, :] = self.o_for_array[n].ra(tf[-1])
            self.dec_last[n, :] = self.o_for_array[n].dec(tf[-1]) 
            self.dist_last[n, :] = self.o_for_array[n].dist(tf[-1]) 
            self.pmra_last[n, :] = self.o_for_array[n].pmra(tf[-1]) 
            self.pmdec_last[n, :] = self.o_for_array[n].pmdec(tf[-1]) 
            self.vlos_last[n, :] = self.o_for_array[n].vlos(tf[-1])
            
    def _get_forward_integration(self):
        """
        Computes the forward integration.

        :param tf: Array of time values for integration.
        :param pot: Gravitational potential used for integration.
        :param N_realization: Number of forward realizations. Defaults to 1.
        :return: Array of Orbit objects for each realization.
        
        """

        self.o_for_array =  np.empty(self.N_realization, dtype=object)
        
        self.v_x = np.zeros((self.N_realization, self.num ))
        self.v_y = np.zeros((self.N_realization, self.num )) 
        self.v_z = np.zeros((self.N_realization, self.num ))
        
        R, phi, z_cyl = coords.rect_to_cyl(self.x*units.kpc,
                                           self.y*units.kpc,
                                           self.z*units.kpc)

    
        for n in range(self.N_realization):
            self.v_x[n,:], self.v_y[n,:], self.v_z[n,:] = self._velocities(self.tan_theta, self.pot)
                        
            vR, vT, vz = coords.rect_to_cyl_vec(
                self.v_x[n,:]*units.km/units.s, 
                self.v_y[n,:]*units.km/units.s, 
                self.v_z[n,:]*units.km/units.s, 
                self.x*units.kpc,
                self.y*units.kpc,
                self.z*units.kpc
                )
            o_for = Orbit([R, vR, vT, z_cyl, vz, phi])
            o_for.integrate(self.tf, self.pot)           
            self.o_for_array[n] = o_for
            
        self._cart_evol(self.tf)
        self._forward_last(self.tf)
        
        
        if isinstance(self.o_for_array, np.ndarray) and self.o_for_array.size == 1:
            return self.o_for_array.item()
        else:
            return self.o_for_array
        
    def _axes_forward(self):

        Nstep = len(self.tf)
        self.for_magA = np.zeros((self.N_realization,  Nstep))
        self.for_magB= np.zeros((self.N_realization,  Nstep))
        self.for_magC = np.zeros((self.N_realization, Nstep))
        for n in range(self.N_realization):
            for t in range(Nstep):
                xpos = self.xfor_evo[n,:,t]
                ypos = self.yfor_evo[n,:,t]
                zpos = self.zfor_evo[n,:,t]
                pos_for = np.vstack([xpos, ypos, zpos]).T
                axes_val = toi(pos_for)
                self.for_magA[n, t] = axes_val["MagA"]
                self.for_magB[n, t] = axes_val["MagB"]
                self.for_magC[n, t] = axes_val["MagC"]
                
    def get_ca_for(self, do_print = True):
        """
        Calculate the axis ratios (c/a) for the forward integration and compute mean and standard deviation.
        """
        if  self.o_for_array is None:
            raise ValueError("forward_integration must be called to get ca.")
        
        self._axes_forward()
        self.ca_for_all = self.for_magC/self.for_magA
        
        self.ca_for_mean = np.mean(self.ca_for_all, axis = 0)
        self.ca_for_std = np.std(self.ca_for_all, axis = 0)
        
        if do_print is True:
            return self.ca_for_all
    
        
    def get_unbound_sat(self, max_r, do_print = False):
        
        Nstep = len(self.tf)
        self.unbound_sat_for_all = np.zeros((self.N_realization, Nstep))
        for n in range(self.N_realization):
            for t in range(Nstep):
                self.unbound_sat_for_all[n,t] =  np.sum(self.o_for_array[n].r(self.tf[t])>max_r)
                
        self.unbound_sat_for_mean = np.mean(self.unbound_sat_for_all, axis =0)
        self.unbound_sat_for_std =  np.std(self.unbound_sat_for_all, axis =0)
        if do_print is True:
            return self.unbound_sat_for_all
        
def toi(vecs: np.ndarray):

    """ Performs unweighted tensor-of-inertia fit to calculate 3D plane properties.
    Parameters:
        pos: array of satellite positions
    Returns:
        toi: dictionary of distribution parameters
    """
    

    assert vecs.ndim == 2, "Requires multiple satellite positions."
    assert vecs.shape[0] > 3, "Need at least 4 satellites for plane-finding."
        
    inertia_tensor = np.zeros((3, 3))

    for vec in vecs:
        vector = np.mat(vec)
        left = float(np.dot(vector, vector.T)) * np.eye(3, 3)
        right = vector.T * vector
        inertia_tensor = inertia_tensor + left - right
    
    eig_vals, eig_vecs = np.linalg.eig(inertia_tensor)  # Eigenvalues and eigenvectors of ToI

    index, norm, mag = {}, {}, {}

    index["a"] = np.argmin(eig_vals)  # Index of major axis
    index["c"] = np.argmax(eig_vals)  # Index of minor axis
    index["b"] = int(np.setdiff1d([0, 1, 2], [index["a"], index["c"]])[0])  # Index of intermediate axis

    for axis in ("a", "b", "c"):

        norm[axis] = np.array(eig_vecs.T[index[axis]])[0]  # Calculate axis orientations

        mag_squared = np.sum(np.array([np.dot(vec, norm[axis]) ** 2 for vec in vecs]))
        mag[axis] = np.sqrt(mag_squared / len(vecs))  # Calculate axis rms magnitudes

    toi = {
        "AxisA": norm["a"], "AxisB": norm["b"], "AxisC": norm["c"],
        "MagA": mag["a"], "MagB": mag["b"], "MagC": mag["c"]
    }

    return toi


        
class BackwardIntegration:
    def __init__(self, forward_int, tb, pot_back,  M_realization = 1, pmerr_list = [0,0], errtype  = "normal", dist_err = 0 ):
        self.forward_int = forward_int

        self.N_realization = self.forward_int.N_realization
        self.num = self.forward_int.num

        self.tb =  tb
        self.M_realization =  M_realization
        self.pmerr_list = pmerr_list
        self.errtype = errtype
        self.dist_err = dist_err
        self.pot_back = pot_back
        
        self.xback_evo = None
        self.yback_evo = None
        self.zback_evo = None
        
        self.o_back_array = None
        
        self.back_magA = None
        self.back_magB = None
        self.back_magC = None
        
        self.ca_back_all = None
        self.ca_back_mean = None
        self.ca_back_std = None

        self._get_backward_integration()

            
    def _proper_motion_errors(self, pmerr, type):
        """
        Generate proper motion errors.

        :param pmerr: List of proper motion errors.
        :param type: Type of error distribution ('normal' or 'uniform').
        :return: Tuple of numpy arrays containing proper motion errors for ra and dec.
        """
        
        if type == "normal" or "Normal":
            pmra_error_vals  = np.random.normal(0, pmerr[0]/3, self.num)
            pmdec_error_vals = np.random.normal(0, pmerr[1]/3, self.num)
        
        elif type == "uniform" or "Uniform":
            pmra_error_vals = np.random.uniform(low=-pmerr[0], high=pmerr[0], size=self.num)
            pmdec_error_vals = np.random.uniform(low=-pmerr[1], high=pmerr[1], size=self.num)
            
        return pmra_error_vals, pmdec_error_vals
    
    
    def _get_backward_integration(self):
        
        
        ra_last =  self.forward_int.ra_last
        dec_last =  self.forward_int.dec_last
        pmra_last =  self.forward_int.pmra_last
        pmdec_last =  self.forward_int.pmdec_last
        dist_last =  self.forward_int.dist_last
        vlos_last =  self.forward_int.vlos_last
        
        self.xback_evo = np.empty((self.N_realization, self.M_realization, self.num, len(self.tb)))
        self.yback_evo = np.empty((self.N_realization, self.M_realization, self.num, len(self.tb)))
        self.zback_evo = np.empty((self.N_realization, self.M_realization, self.num, len(self.tb)))
        
        self.o_back_array = np.empty((self.N_realization, self.M_realization ), dtype=object)
       
        for n in range(self.N_realization):
                               
            for m in range(self.M_realization):
                
                
                pmra_error, pmdec_error = self._proper_motion_errors(self.pmerr_list, self.errtype)
    
                coord = SkyCoord(ra=ra_last[n]*units.deg,
                                    dec=dec_last[n]*units.deg,
                                    distance= (dist_last[n]+(dist_last[n]*(self.dist_err/100.)))*units.kpc,
                                    pm_ra_cosdec=(pmra_last[n]+ pmra_error)*units.mas/units.yr,
                                    pm_dec= (pmdec_last[n]+ pmdec_error)*units.mas/units.yr,
                                    radial_velocity=vlos_last[n]*units.km/units.s)
                
                o_back = Orbit(coord)
                o_back.integrate(self.tb, self.pot_back)
                
                self.o_back_array[n, m] = o_back
                self.xback_evo[n,m,:,:] = o_back.x(self.tb)
                self.yback_evo[n,m,:,:] = o_back.y(self.tb)
                self.zback_evo[n,m,:,:] = o_back.z(self.tb)
                
                
        
        if isinstance(self.o_back_array, np.ndarray) and self.o_back_array.size == 1:
            return self.o_back_array.item()
        else:
            return self.o_back_array
        
        
    def _axes_back(self):
        """
        Calculate the axis (a, b and c) for the backward integration.
        """
        Nstep = len(self.tb)
        self.back_magA = np.zeros((self.N_realization, self.M_realization, Nstep))
        self.back_magB= np.zeros((self.N_realization, self.M_realization, Nstep))
        self.back_magC = np.zeros((self.N_realization, self.M_realization, Nstep))


        for n in range(self.N_realization):
            for m in range(self.M_realization):
                for t in range(Nstep):
                    x = self.xback_evo[n,m,:,t]
                    y = self.yback_evo[n,m,:,t]
                    z = self.zback_evo[n,m,:,t]   
                    pos = np.vstack([x,y,z]).T
                    axesval = toi(pos)
                    self.back_magA[n, m, t] = axesval["MagA"]
                    self.back_magB[n, m, t] = axesval["MagB"]
                    self.back_magC[n, m, t] = axesval["MagC"]
        return {"MagA":self.back_magA, "MagB":self.back_magB, "MagC":self.back_magC}
    

    def get_ca_back(self, do_print = False):
        """
        Calculate the axis ratios (c/a) for the backward integration and compute mean and standard deviation.
        """
        if  self.o_back_array is None:
            raise ValueError("backward_integration must be called to get ca.")
        
        self._axes_back()
        self.ca_back_all = self.back_magC/self.back_magA
        
        self.ca_back_mean = np.mean(self.ca_back_all, axis= (0,1))
        self.ca_back_std = np.std(self.ca_back_all, axis= (0,1))
        if do_print is True:
            return self.ca_back_all
    
    
    def get_unbound_sat(self, max_r, do_print =False):
        Mstep = len(self.tb)
        self.unbound_number_back = np.zeros((self.N_realization, self.M_realization, Mstep))
        for n in range(self.N_realization):
            for m in range(self.M_realization):
                for t in range(Mstep):
                    self.unbound_number_back[n, m, t]  = np.sum(self.o_back_array[n, m].r(self.tb[t]) > max_r)

        self.unbound_sat_back_mean =np.mean(self.unbound_number_back, axis= (0,1))
        self.unbound_sat_back_std =np.std(self.unbound_number_back, axis= (0,1))
        if do_print is True:
            return self.unbound_number_back
