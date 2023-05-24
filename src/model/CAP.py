import numpy as np
from scipy.stats import (dirichlet, invgamma, invwishart, multivariate_normal,
                         multivariate_t, norm, t)
from tqdm import tqdm


class CAP:
    def __init__(
        self,
        alpha: int,
        beta: int,
        b: int,
        gamma: int,
        mu_0: np.ndarray,
        kappa_0: int,
        psi_0: np.ndarray,
        rho_0: int,
        nu_0: np.ndarray,
        lambda_0: int,
        epsilon_0: np.ndarray,
        tau_0: int,
        n_regions: int,
        n_categories: int,
        n_temporal_components: int,
    ):
        self.alpha = alpha
        self.beta = beta
        self.b = b
        self.gamma = gamma
        self.mu_0, self.kappa_0, self.psi_0, self.rho_0 = mu_0, kappa_0, psi_0, rho_0
        self.nu_0, self.lambda_0, self.epsilon_0, self.tau_0 = nu_0, lambda_0, epsilon_0, tau_0

        self.n_regions = n_regions
        self.n_categories = n_categories
        self.n_temporal_components = n_temporal_components

    def _initialize_hidden_variables(self, D_L: np.ndarray, D_U: np.ndarray):
        self.r = np.random.randint(0, self.n_regions, len(D_L) + len(D_U))
        self.c = np.concatenate((D_L[:, 4].astype(int), np.random.randint(0, self.n_categories, len(D_U))))
        self.z = np.random.randint(0, self.n_temporal_components, len(D_L) + len(D_U))

    def _initialize_num_elements(self):
        self.r_num_elements = np.array([np.count_nonzero(self.r == region) for region in range(self.n_regions)])
        self.cz_num_elements = np.zeros((self.n_categories, self.n_temporal_components))
        self.rc_num_elements = np.zeros((self.n_regions, self.n_categories))
        self.ruc_num_elements = np.zeros((self.n_regions, self.n_categories))
        for c, z in zip(self.c, self.z):
            self.cz_num_elements[c, z] += 1
        for r, c in zip(self.r, self.c):
            self.rc_num_elements[r, c] += 1
        for i, (r, c) in enumerate(zip(self.r, self.c)):
            u = self.D[i][0].astype(int)
            if u == self.user:
                self.ruc_num_elements[r, c] += 1

    def _sub_num_elements(self, i: int):
        self.r_num_elements[self.r[i]] -= 1
        self.cz_num_elements[self.c[i], self.z[i]] -= 1
        self.rc_num_elements[self.r[i], self.c[i]] -= 1
        if self.D[i][0].astype(int) == self.user:
            self.ruc_num_elements[self.r[i], self.c[i]] -= 1

    def _add_num_elements(self, i: int):
        self.r_num_elements[self.r[i]] += 1
        self.cz_num_elements[self.c[i], self.z[i]] += 1
        self.rc_num_elements[self.r[i], self.c[i]] += 1
        if self.D[i][0].astype(int) == self.user:
            self.ruc_num_elements[self.r[i], self.c[i]] += 1

    def _update_kappa(self):
        return self.kappa_0 + self.r_num_elements

    def _update_mu(self, l_bar: np.ndarray):
        return (self.kappa_0 * self.mu_0 + self.r_num_elements.reshape((-1, 1)) * l_bar) / self.kappa.reshape(
            (-1, 1)
        )

    def _update_rho(self):
        return self.rho_0 + self.r_num_elements

    def _update_psi(self, l_bar: np.ndarray, l_cov: np.ndarray):
        kappa_term = np.zeros((self.n_regions, 2, 2))
        for r in range(self.n_regions):
            kappa_term[r] = np.outer(l_bar[r] - self.mu_0, l_bar[r] - self.mu_0)
        kappa_term = (self.kappa_0 * self.r_num_elements / self.kappa).reshape(-1, 1, 1) * kappa_term
        return self.psi_0.reshape(1, 2, 2) + l_cov + kappa_term

    def _update_lambda(self):
        return self.lambda_0 + self.cz_num_elements

    def _update_nu(self, t_bar: int):
        return (self.lambda_0 * self.nu_0 + self.cz_num_elements * t_bar) / self.lamb

    def _update_tau(self):
        return self.tau_0 + self.cz_num_elements / 2

    def _update_epsilon(self, t_bar: np.ndarray, t_cov: np.ndarray):
        lambda_term = self.lambda_0 * self.cz_num_elements / (2 * self.lamb) * (t_bar - self.nu_0) ** 2
        return self.epsilon_0 + t_cov / 2 + lambda_term

    def _update_theta(self):
        numerator = self.r_num_elements + self.alpha
        denominator = np.sum([self.r_num_elements[region] + self.alpha for region in range(self.n_regions)])
        return numerator / denominator

    def _update_si(self):
        numerator = self.cz_num_elements + self.gamma
        denominator = np.sum(self.cz_num_elements + self.gamma, axis=1)
        return np.divide(
            numerator, denominator[:, None], out=np.zeros_like(numerator), where=denominator[:, None] != 0
        )

    def _update_phi(self):
        numerator = self.rc_num_elements + self.beta
        denominator = np.sum(self.rc_num_elements + self.beta, axis=1)
        return np.divide(
            numerator, denominator[:, None], out=np.zeros_like(numerator), where=denominator[:, None] != 0
        )

    def _update_phi_u(self):
        numerator = self.ruc_num_elements + self.b * self.phi
        denominator = np.sum(self.ruc_num_elements + self.b * self.phi, axis=1).reshape(-1, 1)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    def _update_parameters(self, i: int):
        l_bar = np.empty((self.n_regions, 2))
        l_cov = np.empty((self.n_regions, 2, 2))
        for region in range(self.n_regions):
            indices = np.where(self.r == region)[0]
            if len(indices) < 1:
                l_bar[region] = np.array([0, 0])
                l_cov[region] = np.array([[0, 0], [0, 0]])
            else:
                l_bar[region] = np.mean(self.D[indices][:, [1, 2]], axis=0)
                l_cov[region] = np.cov(self.D[indices][:, [1, 2]], rowvar=False)
        self.kappa = self._update_kappa()
        self.mu = self._update_mu(l_bar)
        self.rho = self._update_rho()
        self.psi = self._update_psi(l_bar, l_cov)

        t_bar = np.zeros((self.n_categories, self.n_temporal_components))
        t_cov = np.zeros((self.n_categories, self.n_temporal_components))
        for i in range(len(self.D)):
            c, z = self.c[i], self.z[i]
            if z == -1:
                continue
            t_bar[c][z] += self.D[i][3]
            t_cov[c][z] += self.D[i][3] ** 2
        t_bar = np.divide(t_bar, self.cz_num_elements, out=np.zeros_like(t_bar), where=self.cz_num_elements != 0)
        t_cov = np.divide(t_cov, self.cz_num_elements, out=np.zeros_like(t_cov), where=self.cz_num_elements != 0)
        t_cov = np.where(t_cov == 0, 0, t_cov - t_bar**2)
        self.lamb = self._update_lambda()
        self.nu = self._update_nu(t_bar)
        self.tau = self._update_tau()
        self.epsilon = self._update_epsilon(t_bar, t_cov)

        self.theta = self._update_theta()
        self.si = self._update_si()

    def _update_parameters_labeled_data(self, i: int):
        self._update_parameters(i)
        self.phi = self._update_phi()

    def _update_parameters_unlabeled_data(self, i: int):
        self._update_parameters(i)
        self.phi_u = self._update_phi_u()

    def _calculate_posterior_probabilities_labeled_data(self, i: int):
        l_i, t_i, c_i = self.D[i][[1, 2]], self.D[i][3], self.D[i][4].astype(int)
        r_posterior_probabilities = np.zeros(self.n_regions)
        z_posterior_probabilities = np.zeros(self.n_temporal_components)

        for r in range(self.n_regions):
            p_r = multivariate_t.pdf(
                l_i,
                loc=self.mu[r].astype(float),
                shape=self.psi[r].astype(float) * (self.kappa[r] + 1) / (self.kappa[r] * (self.rho[r] - 1)),
                df=self.rho[r] - 1,
            )
            p_r *= self.theta[r] * self.phi[r][c_i]
            r_posterior_probabilities[r] = p_r

        for z in range(self.n_temporal_components):
            # calculate p(z_i=z|d_-i)
            p_z = t.pdf(
                t_i,
                df=2 * self.tau[c_i][z],
                loc=self.nu[c_i][z],
                scale=self.epsilon[c_i][z] * (self.lamb[c_i][z] + 1) / (self.lamb[c_i][z] * self.tau[c_i][z]),
            )
            p_z *= self.si[c_i][z]
            z_posterior_probabilities[z] = p_z

        return r_posterior_probabilities, z_posterior_probabilities

    def _calculate_posterior_probabilities_unlabeled_data(self, i: int):
        l_i, t_i = self.D[i][[1, 2]], self.D[i][3]
        posterior_probabilities = np.zeros((self.n_regions, self.n_categories, self.n_temporal_components))
        r_posterior_probabilities = np.zeros((self.n_regions))
        cz_posterior_probabilities = np.zeros((self.n_categories, self.n_temporal_components))

        for r in range(self.n_regions):
            # calculate p(r_i=r,c_i=c|d_-i)
            p_r = multivariate_t.pdf(
                l_i,
                df=self.rho[r] - 1,
                loc=self.mu[r].astype(float),
                shape=self.psi[r].astype(float) * (self.kappa[r] + 1) / (self.kappa[r] * (self.rho[r] - 1)),
            )
            r_posterior_probabilities[r] = p_r

        for c in range(self.n_categories):
            for z in range(self.n_temporal_components):
                p_cz = t.pdf(
                    t_i,
                    df=2 * self.tau[c][z],
                    loc=self.nu[c][z],
                    scale=self.epsilon[c][z] * (self.lamb[c][z] + 1) / (self.lamb[c][z] * self.tau[c][z]),
                )
                cz_posterior_probabilities[c][z] = p_cz

        for r in range(self.n_regions):
            for c in range(self.n_categories):
                for z in range(self.n_temporal_components):
                    posterior_probabilities[r][c][z] = (
                        r_posterior_probabilities[r]
                        * self.theta[r]
                        * self.phi_u[r][c]
                        * cz_posterior_probabilities[c][z]
                        * self.si[c][z]
                    )

        return posterior_probabilities

    def _sample_new_r_z(self, r_posterior_probabilities: np.ndarray, z_posterior_probabilities: np.ndarray):
        if np.sum(r_posterior_probabilities) != 0:
            r_posterior_probabilities = r_posterior_probabilities / np.sum(r_posterior_probabilities)
        else:
            r_posterior_probabilities = np.ones(len(r_posterior_probabilities)) / len(r_posterior_probabilities)
        if np.sum(z_posterior_probabilities) != 0:
            z_posterior_probabilities = z_posterior_probabilities / np.sum(z_posterior_probabilities)
        else:
            z_posterior_probabilities = np.ones(len(z_posterior_probabilities)) / len(z_posterior_probabilities)
        r = np.random.choice(np.arange(self.n_regions), p=r_posterior_probabilities)
        z = np.random.choice(np.arange(self.n_temporal_components), p=z_posterior_probabilities)
        return r, z

    def _sample_new_r_c_z(self, posterior_probabilities: np.ndarray):
        posterior_probabilities = posterior_probabilities / np.sum(posterior_probabilities)
        flat_index = np.random.choice(np.arange(posterior_probabilities.size), p=posterior_probabilities.ravel())
        r, c, z = np.unravel_index(flat_index, posterior_probabilities.shape)
        return r, c, z

    def collapsed_gibbs_sampling(self, M: int, D_L: np.ndarray, D_U: np.ndarray):
        self.user = D_U[0][0].astype(int)
        self.D = np.vstack((D_L, D_U))
        self._initialize_hidden_variables(D_L, D_U)
        self._initialize_num_elements()

        # For labeled data
        for iter in tqdm(range(M)):
            for i, d_i in enumerate(D_L):
                # Remove d_i from r_i, z_i
                self._sub_num_elements(i)
                self.r[i], self.z[i] = -1, -1
                # Update parameters based on equations (4) - (8)
                self._update_parameters_labeled_data(i)
                # Calculate posterior probabilities and sample new r_i, z_i based on equations (11)
                (
                    r_posterior_probabilities,
                    z_posterior_probabilities,
                ) = self._calculate_posterior_probabilities_labeled_data(i)
                self.r[i], self.z[i] = self._sample_new_r_z(r_posterior_probabilities, z_posterior_probabilities)
                self._add_num_elements(i)
        # For unlabeled data
        for iter in tqdm(range(M)):
            for i, d_i in enumerate(D_U):
                # Remove d_i from r_i, c_i, z_i
                self._sub_num_elements(i + len(D_L))
                self.r[i + len(D_L)], self.c[i + len(D_L)], self.z[i + len(D_L)] = -1, -1, -1
                # Update parameters based on equations (4) - (9)
                self._update_parameters_unlabeled_data(i + len(D_L))
                # Calculate posterior probabilities and sample new r_i, c_i, z_i
                posterior_probabilities = self._calculate_posterior_probabilities_unlabeled_data(i + len(D_L))
                self.r[i + len(D_L)], self.c[i + len(D_L)], self.z[i + len(D_L)] = self._sample_new_r_c_z(
                    posterior_probabilities
                )
                self._add_num_elements(i + len(D_L))

    def _sample_from_niw(self, r: int):
        # Sample from Inverse-Wishart distribution
        covar_matrix = invwishart(df=self.rho[r], scale=self.psi[r]).rvs()
        # Sample from Normal distribution
        mean_vector = multivariate_normal(mean=self.mu[r], cov=covar_matrix / self.kappa[r]).rvs()
        return mean_vector, covar_matrix

    def _sample_from_nig(self, c: int, z: int):
        # Sample from Inverse-Gamma distribution
        sigma = invgamma(a=self.tau[c][z], scale=self.epsilon[c][z]).rvs()
        # Sample from Normal distribution
        mean = norm(loc=self.nu[c][z], scale=np.sqrt(sigma / self.lamb[c][z])).rvs()
        return mean, sigma

    def _calculate_poi_probabilities(self, d_i: np.ndarray, candidate_pois: np.ndarray):
        u_i = d_i[0]
        l_i = d_i[[1, 2]]
        t_i = d_i[3]
        mu = np.zeros((self.n_regions, 2))
        sigma = np.empty((self.n_regions, 2, 2))
        for r in range(self.n_regions):
            mu[r], sigma[r] = self._sample_from_niw(r)

        region_probabilities = [
            self.theta[r] * multivariate_normal.pdf(l_i, mu[r], sigma[r]) for r in range(self.n_regions)
        ]
        region_probabilities /= np.sum(region_probabilities)

        poi_probabilities = np.zeros(len(candidate_pois))
        for c in range(len(candidate_pois)):
            region_term = 0
            temporal_term = 0
            for r in range(self.n_regions):
                region_term += region_probabilities[r] * self.phi_u[r][c]
            for z in range(self.n_temporal_components):
                nu_cz, sigma_cz = self._sample_from_nig(c, z)
                temporal_term += self.si[c][z] * norm.pdf(t_i, loc=nu_cz, scale=sigma_cz)
            poi_probabilities[c] = region_term * temporal_term

        return poi_probabilities

    def annotate_records(self, records: np.ndarray, candidate_pois: np.ndarray):
        annotated_records = []
        for i, d_i in enumerate(records):
            poi_probabilities = self._calculate_poi_probabilities(d_i, candidate_pois[i])
            max_prob_index = np.argmax(poi_probabilities)
            annotated_records.append(candidate_pois[i][max_prob_index])

        return annotated_records

    def generative_process(self, n_users: int, n_records: int):
        theta = dirichlet.rvs(np.ones(self.n_regions) * self.alpha, size=1)[0]
        phi_ru = np.zeros((self.n_regions, n_users, self.n_categories))
        phi_r = np.zeros((self.n_regions, self.n_categories))
        mu = {}
        sigma_r = {}

        for r in range(self.n_regions):
            mu[r], sigma_r[r] = self._sample_NIW()
            phi_r[r, :] = dirichlet.rvs(np.ones(self.n_categories) * self.beta, size=1)[0]
            for u in range(n_users):
                phi_ru[r, u, :] = dirichlet.rvs(self.b * phi_r[r, :], size=1)[0]

        psi_c = np.zeros((self.n_categories, self.n_temporal_components))
        nu = np.zeros((self.n_categories, self.n_temporal_components))
        sigma_cz = np.zeros((self.n_categories, self.n_temporal_components))
        for c in range(self.n_categories):
            psi_c[c, :] = dirichlet.rvs(np.ones(self.n_temporal_components) * self.gamma, size=1)[0]
            for z in range(self.n_temporal_components):
                nu[c, z], sigma_cz[c, z] = self._sample_NIG()

        records = []
        for d in range(n_records):
            r = np.random.choice(self.n_regions, p=theta)
            u = np.random.choice(n_users)
            l = multivariate_normal.rvs(mu[r], sigma_r[r])
            c = np.random.choice(self.n_categories, p=phi_ru[r, u, :])
            z = np.random.choice(self.n_temporal_components, p=psi_c[c, :])
            t = norm.rvs(nu[c, z], sigma_cz[c, z])
            record = {"region": r, "user": u, "location": l, "category": c, "temporal_component": z, "time": t}
            records.append(record)

        return records
