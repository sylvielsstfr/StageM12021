import logging
import numpy as np
import numba as nb
import astropy.units as units
import astropy.constants as constants

#Model parameters
a = 5000	#core radius
rho_0 = 0.0079 	#dark matter density near sun
d_sol = 8500	#distance between the galactic center and the sun
vrot_sol = 239 # km/s 	#rotation speed of the sun
l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
r_lmc = 55000 #LMC dista,ce
r_earth = (150*1e6*units.km).to(units.pc).value  #Earth orbit radius

#Halo parameters
sigma_h = 120 #halo dark matter velocity dispersion  from https://doi.org/10.1111/j.1365-2966.2005.09367.x

pc_to_km = (units.pc.to(units.km))
kms_to_pcd = (units.km/units.s).to(units.pc/units.d)

cosb_lmc = np.cos(b_lmc)
cosl_lmc = np.cos(l_lmc)
A = d_sol ** 2 + a ** 2
B = d_sol * cosb_lmc * cosl_lmc
r_0 = np.sqrt(4*constants.G/(constants.c**2)*r_lmc*units.pc).decompose([units.Msun, units.pc]).value

epsilon = (90. - 66.56070833)*np.pi/180. #ecliptic tilt
delta_lmc = -69.756111 * np.pi/180.      #LMC direction
alpha_lmc = 80.89417 * np.pi/180.


def cartgal_to_heliosphgal(vx, vy, vz):
	"""Transform cartesian galactocentric coordinates to heliocentric galactic coordinates
	cartesian galactocentric defined by :
		origin in galactic center
		x toward the sun
		y in the rotation direction of the Sun (anti-trigo in regard to the galactic North Pole)
		z toward the galactic North Pole

	heliocentric galactic defined by:
		origin at the Sun
		x toward the galactic center
		z toward the galactic north pole
		y to make the referential direct
	"""
	v = np.array([vx, vy, vz])
	rot1 = np.array([
		[np.cos(l_lmc), np.sin(l_lmc), 0],
		[-np.sin(l_lmc), np.cos(l_lmc), 0],
		[0, 0, 1]
	])

	rot2 = np.array([
		[np.cos(b_lmc), 0, np.sin(b_lmc)],
		[0, 1, 0],
		[-np.sin(b_lmc), 0, np.cos(b_lmc)]
	])

	# print(v)
	# print(rot1 @ v)
	# print(rot2 @ rot1 @ v)
	return rot2 @ rot1 @ v


v_lmc = cartgal_to_heliosphgal(-57, -226, 221)
# Compute LMC speed vector in heliospherical galactic coordinates
v_sun = cartgal_to_heliosphgal(11.1, 12.24 + vrot_sol, 7.25)
# Compute speed vector of the Sun in heliospherical galactic coordinates (particular speed + global rotation speed)


def compute_i():
	""" Compute the i vector coordinates, in heliopsheric galactic coordinates.
	i is the vector of the projected plane referential."""

	rot1qc = np.array([
		[1, 0, 0],
		[0, np.cos(epsilon), np.sin(epsilon)],
		[0, -np.sin(epsilon), np.cos(epsilon)]
	])

	def eq_to_ec(v):
		return rot1qc @ np.array(v)

	def ec_to_eq(v):
		return rot1qc.T @ np.array(v)

	rotYlmc = np.array([
		[np.cos(delta_lmc), 0, np.sin(delta_lmc)],
		[0, 1, 0],
		[-np.sin(delta_lmc), 0, np.cos(delta_lmc)]
	])
	rotZlmc = np.array([
		[np.cos(alpha_lmc), np.sin(alpha_lmc), 0],
		[-np.sin(alpha_lmc), np.cos(alpha_lmc), 0],
		[0, 0, 1]
	])
	def eq_to_lmc(v):
		return rotYlmc @ rotZlmc @ v

	K = eq_to_lmc(ec_to_eq([0, 0, 1]))
	i = np.cross(K, np.array([1, 0, 0]))
	i = i/np.linalg.norm(i)
	return i


@nb.njit
def project_from_gala(vr, vtheta, vz, x):
	"""Project speed vector located on the LoS toward the LMC, at x, in heliospherical galactic coordinates.
	Only returns components orthogonal to the LoS

	Parameters
	----------
	vr, vtheta, vz : float
		speed vector coordinates in heliospherical galactic coordinates
	x : float
		distance ratio to LMC

	Returns
	-------
	vgala_theta, vgala_phi : (float, float)
			theta and phi components of projected speed vector orthogonal to LoS, in heliospherical galactic coordinates
	"""
	r = np.sqrt((x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol) ** 2 + (x * r_lmc * np.cos(b_lmc) * np.sin(l_lmc)) ** 2)
	sin_theta = (x * r_lmc * np.sin(l_lmc) * np.cos(b_lmc))
	cos_theta = (x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol)
	theta = np.arctan2(sin_theta, cos_theta)
	cosa = np.cos(theta - l_lmc)
	sina = np.sin(theta - l_lmc)

	vhelio_r = vr * cosa - vtheta * sina
	vhelio_theta = vr * sina + vtheta * cosa
	vhelio_z = vz

	# vgala_r = np.cos(b_lmc) * vhelio_r + np.sin(b_lmc) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_lmc) * vhelio_r + np.cos(b_lmc) * vhelio_z

	return vgala_theta, vgala_phi


def compute_thetas(vr, vtheta, vz, x):
	v = project_from_gala(vr, vtheta, vz, x)
	v = np.array([np.zeros(len(v[0])), *v])
	i = compute_i()
	thetas = np.arctan2(i[1]*v[2]-i[2]*v[1], i[1]*v[1]+i[2]*v[2])
	return thetas


@nb.njit
def vt_from_vs(vr, vtheta, vz, x):
	"""Transform speed vector located on the LoS toward the LMC, at x, in heliospherical galactic coordinates,
	then project it on the plane orthogonal to the LoS and returns the norm

	Parameters
	----------
	vr, vtheta, vz : float
		speed vector coordinates in heliospherical galactic coordinates
	x : float
		distance ratio to LMC

	Returns
	-------
	vt : float
		norm of speed vector orthogonally projected to LoS
	"""
	#r = np.sqrt((x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol) ** 2 + (x * r_lmc * np.cos(b_lmc) * np.sin(l_lmc)) ** 2)
	sin_theta = (x * r_lmc * np.sin(l_lmc) * np.cos(b_lmc))
	cos_theta = (x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol)
	theta = np.arctan2(sin_theta, cos_theta)
	cosa = np.cos(theta - l_lmc)
	sina = np.sin(theta - l_lmc)

	# logging.debug(vr, vtheta, vz, r)

	vhelio_r = vr * cosa - vtheta * sina
	vhelio_theta = vr * sina + vtheta * cosa
	vhelio_z = vz

	# logging.debug(vhelio_r, vhelio_theta, vhelio_z)

	# vgala_r = np.cos(b_lmc) * vhelio_r + np.sin(b_lmc) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_lmc) * vhelio_r + np.cos(b_lmc) * vhelio_z

	# logging.debug(vgala_r, vgala_theta, vgala_phi, theta*180/np.pi)

	vt = np.sqrt((vgala_theta - v_sun[1] * (1 - x) - v_lmc[1] * x) ** 2 + (vgala_phi - v_sun[2] * (1 - x) - v_lmc[2] * x) ** 2)
	# vt = np.sqrt((vgala_theta - v_sun[1]*(1-x))**2 + (vgala_phi - v_sun[2]*(1-x))**2)		# Do not take the LMC speed into account
	# vt= np.sqrt(vgala_theta**2 + vgala_phi**2)											# Only take the speed of the deflector into account
	return vt  # , vgala_r, vgala_theta, vgala_phi


@nb.njit
def delta_u_from_x(x, mass):
	return r_earth/(r_0*np.sqrt(mass)) * np.sqrt((1-x)/x)


@nb.njit
def tE_from_xvt(x, vt, mass):
	return r_0 * np.sqrt(mass*x*(1-x)) / (vt*kms_to_pcd)


@nb.njit
def rho_halo(x):
	"""Halo dark matter density"""
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)


@nb.njit
def p_v_halo(vr, vtheta, vz):
	"""Particular speed vector probability distribution in halo"""
	v = np.sqrt(vr**2 + vtheta**2 + vz**2)
	return 4*np.pi*v**2 * np.power(2*np.pi*sigma_h**2, -3./2.) * np.exp(-v**2 /(2*sigma_h**2))


@nb.njit
def pdf_xvs_halo(vec):
	"""Disk geometry probabilty density function

	Parameters
	----------

	Returns
	-------
	float
		pdf of (x, vr, vtheta, vz) for the dark matter halo, toward LMC

	"""
	x, vr, vtheta, vz = vec
	if x<0 or x>1:
		return 0		# x should be in [0, 1]
	return np.sqrt(x*(1-x)) * p_v_halo(vr, vtheta, vz) * rho_halo(x) * np.abs(vt_from_vs(vr, vtheta, vz, x))


@nb.njit
def hc_randomizer_halo_LMC(x):
	""" x and vr, vtheta, vz randomizer"""
	scales = [0.13, 160., 160., 160.]
	return np.array([np.random.normal(loc=x[0], scale=scales[0]),
					 np.random.normal(loc=x[1], scale=scales[1]),
					 np.random.normal(loc=x[2], scale=scales[2]),
					 np.random.normal(loc=x[3], scale=scales[3])])


@nb.njit
def metropolis_hastings(func, g, nb_samples, x0, burnin=10000, *args):
	"""
	Metropolis-Hasting algorithm to pick random value following the joint probability distribution func
	Print the number of accepted parameters sets and ratio accepted/refused. This ratio should ideally be around 1/3.

	Parameters
	----------
	func : function
		 Joint probability distribution
	g : function
		Randomizer. Choose it wisely to converge quickly and have a smooth distribution
	nb_samples : int
		Number of points to return. Need to be large so that the output distribution is smooth
	x0 : array-like
		Initial point
	burnin : int
		Number of early generated value to drop (to avoid bias)
	args :
		arguments to pass to *func*

	Returns
	-------
	np.array
		Array containing all the points
	"""
	samples = np.empty((nb_samples+burnin, len(x0)))
	current_x = x0
	accepted=0
	rds = np.random.uniform(0., 1., nb_samples+burnin)			# We generate the rs beforehand, for SPEEEED
	for idx in range(nb_samples+burnin):
		proposed_x = g(current_x)
		tmp = func(current_x, *args)
		if tmp!=0:
			threshold = min(1., func(proposed_x, *args) / tmp)
		else:
			threshold = 1
		if rds[idx] < threshold:
			current_x = proposed_x
			accepted+=1
		samples[idx] = current_x
	print(accepted, accepted/nb_samples)
	# We crop the hundred first to avoid outliers from x0
	return samples[burnin:]


class RealisticGenerator:
	"""
	Class to generate microlensing paramters

	Parameters
	----------
	xvts : str or int
		If a str : Path to file containing x, v_T and theta generated through the Hasting-Metropolis algorithm
		If int, number of x, v_T and theta triplets to sample from (should be at least 10-100 times higher than the number of sets of parameters to generate)
	seed : int
		Seed used for numpy, for reproductibility
	tmin : float
		lower limit of t_0
	tmax : float
		upper limits of t_0
	u_max : float
		maximum u_0
	"""
	def __init__(self, xvts=1e6, seed=None, tmin=48928., tmax=52697., u_max=2.):
		self.seed = seed
		self.xvts = xvts
		self.rdm = np.random.RandomState(seed)
		self.tmin = tmin
		self.tmax = tmax
		self.u_max = u_max
		self.generate_mass = False

		if isinstance(self.xvts, str):
			try:
				self.xvts = np.load(self.xvts)
			except FileNotFoundError:
				logging.error(f"xvt file not found : {self.xvts}")
		elif isinstance(self.xvts, int):
			logging.info(f"Generating {self.xvts} x-vt pairs... ")
			x, vr, vtheta, vz = np.array(metropolis_hastings(pdf_xvs_halo, hc_randomizer_halo_LMC, self.xvts, np.array([0.5, 100., 100., 100.]))).T
			self.xvts = np.array([x, vt_from_vs(vr, vtheta, vz, x), compute_thetas(vr, vtheta, vz, x)])
		else:
			logging.error(f"xvts can't be loaded or generated, check 'xvts' variable : {self.xvts}")

	def generate_parameters(self, mass=30., nb_parameters=1, t0_ranges=None):
		"""
		Generate a set of microlensing parameters, including parallax and blending using S-model and fixed mass

		Parameters
		----------
		mass : float
			mass, in solar masses, for which we generate paramters (\implies \delta_u, t_E)
		nb_parameters : int
			number of parameters set to generate
		t0_ranges : array-like
			individual t0 ranges, if used, should be of length nb_parameters (format : [[tmin_1, ..., tmin_n], [tmax_1, ..., tmax_n]])
		Returns
		-------
		dict
			Dictionnary of lists containing the parameters set:
				u0 is the impact parameter
				t0 is the time of minimum approach between the lens and the Sun-source line.
				tE is the Einstein time (it can be negative or positive)
				delta_u is the projected earth orbit radius in R_E unit
				theta is the angle of the projected trajectory of the lens
				x is the ratio between Sun-lens distance and Sun-LMC distance
				vt is the transverse speed of the lens

		"""
		if self.generate_mass:
			mass = self.rdm.uniform(1, 1000, size=nb_parameters)
		else:
			mass = np.array([mass]*nb_parameters)
		u0 = self.rdm.uniform(0, self.u_max, size=nb_parameters)
		x, vt, theta = (self.xvts.T[self.rdm.randint(0, self.xvts.shape[1], size=nb_parameters)]).T
		vt *= self.rdm.choice([-1., 1.], size=nb_parameters, replace=True)
		delta_u = delta_u_from_x(x, mass=mass)
		tE = tE_from_xvt(x, vt, mass=mass)
		if not t0_ranges is None:
			t0 = self.rdm.uniform(np.array(t0_ranges[0])-2*abs(tE), np.array(t0_ranges[1])+2*abs(tE), size=nb_parameters)
			tmins = np.array(t0_ranges[0])
			tmaxs = np.array(t0_ranges[0])
		else:
			t0 = self.rdm.uniform(self.tmin-2*abs(tE), self.tmax+2*abs(tE), size=nb_parameters)
			tmins = np.array([self.tmin] * nb_parameters)
			tmaxs = np.array([self.tmax] * nb_parameters)
		params = {
			'u0': u0,
			't0': t0,
			'tE': tE,
			'delta_u': delta_u,
			'theta': theta,
			'mass': mass,
			'x': x,
			'vt': vt,
			'tmin' : tmins,
			'tmax' : tmaxs
		}
		return params

#rg = RealisticGenerator(xvts=1000000)
#print(rg.generate_parameters(mass=100, nb_parameters=1000))
