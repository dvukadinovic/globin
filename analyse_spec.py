import rh
import matplotlib.pyplot as plt

spec = rh.Rhout()

spec.read_spectrum("spectrum_tau_scale.out")
spec.read_ray("spectrum_tau_mu1.out")
plt.plot(spec.wave, spec.int)

spec.read_spectrum("spectrum_kappa_scale.out")
spec.read_ray("spectrum_kappa_mu1.out")
plt.plot(spec.wave, spec.int)

plt.show()