import tigre
import tigre.algorithms as algs
import numpy as np, torch
from tigre.utilities import CTnoise


class TigreNoiser:
    def __init__(self, projections, dose=1, _sigma=0, _lambda=1e5, use_dose=True):
        self.projections = projections
        if isinstance(projections, torch.Tensor):
            self.projections = projections.cpu().numpy()
            self.device = projections.device
        else: self.device = None

        if dose < 1:
            self.dose = dose
            self.sigma = _sigma
            self.lambda_ = _lambda

            p_ = np.exp(-self.projections / self.projections.max())
            N0 = 1 / p_[:, :round(0.05 * p_.shape[1]), :round(0.05 * p_.shape[2])].var()
            A = p_.var() / p_.mean()
            Ne = 5
            if use_dose:
                self.lambda_ = (dose / (1 - dose))* N0 / A
                self.sigma = np.sqrt((1 - dose) / dose * p_.mean() / N0 * (1 + (1 + dose) / dose * Ne * p_.mean() / N0))

            # self.noisy_projections = CTnoise.add_dose2(self.projections, Poisson=np.array([(dose/(1-dose))*1e5, (dose/(1-dose))*1e5]), Gaussian=np.array([0, Dc]))
            self.noisy_projections = CTnoise.add(self.projections, Poisson=self.lambda_, Gaussian=np.array([0, self.sigma]))

            if self.device is not None:
                self.noisy_projections = torch.from_numpy(self.noisy_projections).to(self.device)
        else:
            self.noisy_projections = projections


class AngleNoiser:
    def __init__(self, angles, pi=0):
        self.angles = angles
        if pi > 0:
            self.noisy_angles = []
            for angle in angles:
                noise = np.random.uniform(-pi, pi)
                self.noisy_angles.append((angle + round(noise * 180 / np.pi) + 360) % 360)
        else:
            self.noisy_angles = angles