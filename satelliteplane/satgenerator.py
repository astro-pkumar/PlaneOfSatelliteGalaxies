
import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

class SatelliteCoordinateGenerator:
    """
    A class for generating random satellite positions in various coordinate systems.

    :param radi: A list [r_min, r_max] defining the range of radial values.
    :type radi: list
    :param h: A list [mean, std_dev] defining the mean and standard deviation for height values.
    :type h: list
    :param phi: A list [phi_min, phi_max] defining the range of azimuthal angle (phi) values in radians.
    :type phi: list
    :param num: The number of random positions to generate.
    :type num: int

    Attributes:
        num (int): The number of random positions to generate.
        h (list): A list [mean, std_dev] defining the mean and standard deviation for height values.
        phi (list): A list [phi_min, phi_max] defining the range of azimuthal angle (phi) values in radians.
        radi (list): A list [r_min, r_max] defining the range of radial values.
        rad_values (numpy.ndarray): Randomly generated radial values.
        height_values (numpy.ndarray): Randomly generated height values.
        phi_values (numpy.ndarray): Randomly generated azimuthal angle (phi) values in radians.
    """
    
    def __init__(self, radi, h, phi, num):
        
        """
        Initialize the SatelliteCoordinateGenerator instance.

        :param radi: A list [r_min, r_max] defining the range of radial values.
        :type radi: list
        :param h: A list [mean, std_dev] defining the mean and standard deviation for height values.
        :type h: list
        :param phi: A list [phi_min, phi_max] defining the range of azimuthal angle (phi) values in radians.
        :type phi: list
        :param num: The number of random positions to generate.
        :type num: int
        """
        # Input validation checks
        if not isinstance(radi, list) or len(radi) != 2 or not all(isinstance(x, (int, float)) for x in radi):
            raise ValueError(f"`radi` must be a list [r_min, r_max] of two numeric values, but {len(radi)} given.")
        if not isinstance(h, list) or len(h) != 2 or not all(isinstance(x, (int, float)) for x in h):
            raise ValueError(f"`h` must be a list [mean, std_dev] of two numeric values, but {len(h)} given..")
        if not isinstance(phi, list) or len(phi) != 2 or not all(isinstance(x, (int, float)) for x in phi):
            raise ValueError(f"phi` must be a list [phi_min, phi_max] of two numeric values, but {len(phi)} given.")
        if not isinstance(num, int) or num <= 0:
            raise ValueError("`num` must be a positive integer.")
        
        self.num = num
        
        self.h = h
        self.phi = phi
        self.radi = radi
        self.rad_values = None
        self.height_values = None
        self.phi_values = None
        self.x = None
        self.y = None
        self.z = None
        
    def __repr__(self):
        return (f"{self.__class__.__name__}(radi={self.radi}, h={self.h}, "
            f"phi={self.phi}, num={self.num})")

    def _random_rad(self, exponent):
        """
        Generate random radial coordinates based on the specified radial density distribution.

        :param exponent: Exponent for the radial density distribution function.
        :type exponent: float
        :return: Randomly generated radial coordinates.
        :rtype: numpy.ndarray
        """

        # Define the radial density distribution function
        f = lambda x: np.power(x, exponent)

        # Integrate to get the cumulative mass distribution
        fb = lambda x: f(x) * 4.0 * np.pi * np.power(x, 2.0)
        f2 = lambda x: integrate.quad(fb, self.radi[0], x)[0]
        radbins_integral = np.linspace(self.radi[0], self.radi[1], 1001)
        
        value_integral = []
        for i in range(len(radbins_integral)):
            value_integral.append(f2(radbins_integral[i]) / f2(self.radi[1]))

        # Create an interpolation function to invert the cumulative mass distribution
        f_interp = interpolate.interp1d(x=value_integral, y=radbins_integral)
        return f_interp(np.random.uniform(0.0, 1.0, self.num))

    def _random_height(self):
        """
        Generate random height coordinates.

        :return: Randomly generated height coordinates.
        :rtype: numpy.ndarray
        """
        return np.random.normal(self.h[0], self.h[1], self.num)

    def _random_phi(self):
        """
        Generate random azimuthal angle (phi) coordinates.

        :return: Randomly generated azimuthal angle (phi) coordinates in radians.
        :rtype: numpy.ndarray
        """
        return np.random.uniform(self.phi[0], self.phi[1], self.num)
    
    def _rotate_positions(self, theta):
        """
        Rotate the generated positions by a specified angle theta.

        :param theta: The angle in radians by which to rotate the positions.
        :type theta: float

        :return: Rotated positions.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """
        if self.rad_values is None or self.height_values is None or self.phi_values is None:
            raise ValueError("Random values not generated. `Call generate_random_values()` first.")

        x = self.rad_values * np.cos(self.phi_values)
        y = self.rad_values * np.sin(self.phi_values)
        z = self.height_values
        pos = np.vstack((x, y, z)).T

        M = np.array([
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]
            ])
        pos_rotated = (np.dot(M, pos.T)).T
        return pos_rotated
    
    def _plot_params(self):
        return {
                # Figure and axes title/label sizes
                "figure.titlesize": 20,
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                
                # Major Tick settings
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "xtick.major.size": 8,
                "ytick.major.size": 8,
                "xtick.direction": "in",  # Major tick direction: inside
                "ytick.direction": "in",  # Major tick direction: inside
                "xtick.top": True,        # Major ticks on top
                "ytick.right": True,      # Major ticks on right

                # Minor Tick settings
                "xtick.minor.visible": True,  # Display minor ticks
                "ytick.minor.visible": True,  # Display minor ticks
                "xtick.minor.size": 4,        # Size of minor ticks
                "ytick.minor.size": 4,        # Size of minor ticks

                # Axes settings
                "axes.linewidth": 1.5,        # Thickness of the axes

                # Legend settings
                "legend.fontsize": 14,
                "legend.frameon": True,       # Display legend frame

                # Plot settings
                "lines.linewidth": 2,
                "lines.markersize": 8,
                "patch.edgecolor": "black",

                # Color map settings
                "image.cmap": "jet",

                # Other settings
                "figure.figsize": (12, 6),    # Default figure size
                "figure.dpi": 300             # Default figure resolution
            }
    
    def generate_random_values(self, exponent = -3):
        """
        Generate random values for radial, height, and azimuthal angle (phi) coordinates.

        :param exponent: Exponent for the radial density distribution function. Defaults to -3.
        :type exponent: float
        """
        self.rad_values = self._random_rad(exponent)
        self.height_values = self._random_height()
        self.phi_values = self._random_phi()

    def get_random_rad(self):
        """
        Get the randomly generated radial coordinates.

        :return: Randomly generated radial coordinates.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """

        if self.rad_values is None:
            raise ValueError("Random values not generated. Call `generate_random_values()` first.")
        return self.rad_values

    def get_random_height(self):
        """
        Get the randomly generated height coordinates.

        :return: Randomly generated height coordinates.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """
        if self.height_values is None:
            raise ValueError("Random values not generated. Call `generate_random_values()` first.")
        return self.height_values

    def get_random_phi(self):
        """
        Get the randomly generated azimuthal angle (phi) coordinates.

        :return: Randomly generated azimuthal angle (phi) coordinates in radians.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """
        if self.phi_values is None:
            raise ValueError("Random values not generated. Call `generate_random_values()` first.")
        return self.phi_values
    
    
    
    def get_cartesian_coordinates(self, theta=None):
        """
        Convert the generated positions to Cartesian coordinates, optionally applying rotation.
        Stores the coordinates in self.x, self.y, and self.z.

        :param theta: The angle in radians by which to rotate the positions. If None, no rotation is applied.
        :type theta: float, optional

        :return: Cartesian coordinates of the positions.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """
        if self.rad_values is None or self.height_values is None or self.phi_values is None:
            raise ValueError("Random values not generated. Call `generate_random_values()` first.")
        
        if theta is not None:
            pos_rotated = self._rotate_positions(theta)
            self.x, self.y, self.z = pos_rotated[:, 0], pos_rotated[:, 1], pos_rotated[:, 2]
        else:
            self.x = self.rad_values * np.cos(self.phi_values)
            self.y = self.rad_values * np.sin(self.phi_values)
            self.z = self.height_values

        return np.vstack((self.x, self.y, self.z)).T
        
    
    def get_cylindrical_coordinates(self, theta = None):
        """
        Get the cylindrical coordinates of the generated positions, optionally after rotation.

        :param theta: The angle in radians by which to rotate the positions. If None, no rotation is applied.
        :type theta: float, optional

        :return: Cylindrical coordinates of the positions.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """
        if self.rad_values is None or self.height_values is None or self.phi_values is None:
            raise ValueError("Random values not generated. `Call generate_random_values()` first.")
        
        if theta is not None:
            
            # Perform the rotation
            pos_rotated = self._rotate_positions(theta)

            # Convert back to cylindrical coordinates
            r_rotated = np.sqrt(pos_rotated[:, 0]**2 + pos_rotated[:, 2]**2)
            h_rotated = pos_rotated[:, 1]
            phi_rotated = np.arctan2(pos_rotated[:, 2], pos_rotated[:, 0])

            return np.vstack((r_rotated, h_rotated, phi_rotated)).T
        else:
            return np.vstack((self.rad_values, self.height_values, self.phi_values)).T
        
    def get_spherical_coordinates(self, theta =None):
        """
        Get the spherical coordinates of the generated positions, optionally after rotation.

        :param theta: The angle in radians by which to rotate the positions. If None, no rotation is applied.
        :type theta: float, optional

        :return: Spherical coordinates of the positions.
        :rtype: numpy.ndarray

        :raises ValueError: If random values are not generated, call generate_random_values() first.
        """
        if self.rad_values is None or self.height_values is None or self.phi_values is None:
            raise ValueError("Random values not generated. Call `generate_random_values()` first.")

        if theta is not None:
                        
            pos_rotated = self._rotate_positions(theta)
            rho_sph  = np.linalg.norm(pos_rotated, axis=1)
            theta_sph  = np.arccos(pos_rotated[:,2] / rho_sph )
            phi_sph  = np.arctan2(pos_rotated[:,1], pos_rotated[:,0])
            
            return np.vstack((rho_sph, theta_sph, phi_sph)).T            
        
        else:
            rho_sph = np.sqrt(self.rad_values**2 + self.height_values**2)
            phi_sph = np.arctan2(self.rad_values**2, self.height_values**2)
            theta_sph = self.phi_values
            return np.vstack((rho_sph, theta_sph, phi_sph)).T
    
    def plot(self, save_path=None, color_map="jet", marker_size=30, line_width=1.0):
        """
        Plot satellite positions in 1x2 subplots: ZY plot and XY plot.

        :param save_path: Path to save the plot. If None, the plot is not saved.
        :type save_path: str, optional
        :param color_map: Color map for the points.
        :type color_map: str, optional
        :param marker_size: Size of the marker.
        :type marker_size: int, optional
        :param line_width: Line width for the points.
        :type line_width: float, optional
        """
        if self.x is None or self.y is None or self.z is None:
            raise ValueError("Cartesian coordinates not set. Call `get_cartesian_coordinates()` first.")

        # Plotting logic
        #plt.rcParams.update(self._plot_params())
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

        # ZY plot
        axs[0].scatter(self.z, self.y, edgecolor='black',zorder=15, lw=1., c = range(0, self.num), cmap = "jet")
        axs[0].plot([0,0],[0,0] , color = "black", lw=6., marker= "P", markersize =10)

        axs[0].set_xlabel('z  [kpc]')
        axs[0].set_ylabel('y  [kpc]')
        axs[0].set_xlim([-self.radi[1]*1.5, self.radi[1]*1.5])
        axs[0].set_ylim([-self.radi[1]*1.5, self.radi[1]*1.5])

        # XY plot
        axs[1].scatter(self.x, self.y, edgecolor='black',zorder=15, lw=1., c = range(0, self.num), cmap = "jet")
        axs[1].plot([0,0],[0,0] , color = "black", lw=6., marker= "P", markersize =10)
        axs[1].set_xlim([-self.radi[1]*1.5, self.radi[1]*1.5])
        axs[1].set_ylim([-self.radi[1]*1.5, self.radi[1]*1.5])
        axs[1].set_xlabel('x [kpc]')
        axs[1].set_ylabel('y  [kpc]')
        fig.tight_layout()

        plt.show()
        if save_path:
            fig.savefig(f"{save_path}", format = "pdf", bbox_inches = "tight", dpi = 300)
            
            
    