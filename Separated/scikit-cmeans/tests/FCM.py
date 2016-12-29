from scipy.spatial.distance import cdist
import numpy as np
from sklearn.utils import check_random_state


def initialize_random(x, k, random_state=None, eps=1e-12):
    """Selects initial points randomly from the data.
    Parameters
    ----------
    x : :class:`np.ndarray`
        (n_samples, n_features)
        The original data.
    k : int
        The number of points to select.
    random_state : int or :class:`np.random.RandomState`, optional
        The generator used for initialization. Using an integer fixes the seed.
    Returns
    -------
    Unitialized memberships
    selection : :class:`np.ndarray`
        (k, n_features)
        A length-k subset of the original data.
    """
    n_samples = x.shape[0]
    seeds = check_random_state(random_state).permutation(n_samples)[:k]
    selection = x[seeds] + eps
    distances = cdist(x, selection)
    normalized_distance = distances / np.sum(distances, axis=1)[:, np.newaxis]
    return 1-normalized_distance, selection


"""Implementations of a number of C-means algorithms.
References
----------
.. [1] J. C. Bezdek, J. Keller, R. Krisnapuram, and N. R. Pal, Fuzzy models
   and algorithms for pattern recognition and image processing. Kluwer Academic
   Publishers, 2005.
"""

class CMeans:
    """Base class for C-means algorithms.
    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters to find.
    n_init : int, optional
        The number of times to attempt convergence with new initial centroids.
    max_iter : int, optional
        The number of cycles of the alternating optimization routine to run for
        *each* convergence.
    tol : float, optional
        The stopping condition. Convergence is considered to have been reached
        when the objective function changes less than `tol`.
    verbosity : int, optional
        The verbosity of the instance. May be 0, 1, or 2.
        .. note:: Very much not yet implemented.
    random_state : :obj:`int` or :obj:`np.random.RandomState`, optional
        The generator used for initialization. Using an integer fixes the seed.
    eps : float, optional
        To avoid numerical errors, zeros are sometimes replaced with a very
        small number, specified here.
    Attributes
    ----------
    metric : :obj:`string` or :obj:`function`
        The distance metric used. May be any of the strings specified for
        :obj:`cdist`, or a user-specified function.
    initialization : function
        The method used to initialize the cluster centers.
    centers : :obj:`np.ndarray`
        (n_clusters, n_features)
        The derived or supplied cluster centers.
    memberships : :obj:`np.ndarray`
        (n_samples, n_clusters)
        The derived or supplied cluster memberships.
    """

    metric = 'euclidean'
    initialization = staticmethod(initialize_random)

    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4,
                 verbosity=0, random_state=None, eps=1e-18, **kwargs):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbosity = verbosity
        self.random_state = random_state
        self.eps = eps
        self.params = kwargs
        self.centers = None
        self.memberships = None
        if 'metric' in kwargs:
            self.metric = kwargs['metric']

    def distances(self, x):
        """Calculates the distance between data x and the centers.
        The distance, by default, is calculated according to `metric`, but this
        method should be overridden by subclasses if required.
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        Returns
        -------
        :obj:`np.ndarray`
            (n_samples, n_clusters)
            Each entry (i, j) is the distance between sample i and cluster
            center j.
        """
        return cdist(x, self.centers, metric=self.metric)

    def calculate_memberships(self, x):
        raise NotImplementedError(
            "`calculate_memberships` should be implemented by subclasses.")

    def calculate_centers(self, x):
        raise NotImplementedError(
            "`calculate_centers` should be implemented by subclasses.")

    def objective(self, x):
        raise NotImplementedError(
            "`objective` should be implemented by subclasses.")

    def fit(self, x):
        """Optimizes cluster centers by restarting convergence several times.
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        """
        objective_best = np.infty
        memberships_best = None
        centers_best = None
        j_list = []
        for i in range(self.n_init):
            self.centers = None
            self.memberships = None
            self.converge(x)
            objective = self.objective(x)
            j_list.append(objective)
            if objective < objective_best:
                memberships_best = self.memberships.copy()
                centers_best = self.centers.copy()
                objective_best = objective
        self.memberships = memberships_best
        self.centers = centers_best
        return j_list

    def converge(self, x):
        """Finds cluster centers through an alternating optimization routine.
        Terminates when either the number of cycles reaches `max_iter` or the
        objective function changes by less than `tol`.
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        """
        centers = []
        j_new = np.infty
        for i in range(self.max_iter):
            j_old = j_new
            self.update(x)
            centers.append(self.centers)
            j_new = self.objective(x)
            if np.abs(j_old - j_new) < self.tol:
                break
        return np.array(centers)

    def update(self, x):
        """Updates cluster memberships and centers in a single cycle.
        If the cluster centers have not already been initialized, they are
        chosen according to `initialization`.
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        """
        self.initialize(x)
        self.memberships = self.calculate_memberships(x)
        self.centers = self.calculate_centers(x)

    def initialize(self, x):
        if self.centers is None and self.memberships is None:
            self.memberships, self.centers = \
                self.initialization(x, self.n_clusters, self.random_state)
        elif self.memberships is None:
            self.memberships = \
                self.initialization(x, self.n_clusters, self.random_state)[0]
        elif self.centers is None:
            self.centers = \
                self.initialization(x, self.n_clusters, self.random_state)[1]


class FCM(CMeans):
    """Base class for fuzzy C-means clusters.
    Attributes
    ----------
    m : float
        Fuzziness parameter. Higher values reduce the rate of drop-off from
        full membership to zero membership.
    Methods
    -------
    fuzzifier(memberships)
        Fuzzification operator. By default, for memberships $u$ this is $u^m$.
    objective(x)
        Interpretable as the data's weighted rotational inertia about the
        cluster centers. To be minimised.
    """

    m = 2

    def __init__(self, *args, **kwargs):
        super(FCM, self).__init__(*args, **kwargs)
        if 'm' in kwargs:
            self.m = kwargs['m']

    def fuzzifier(self, memberships):
        return np.power(memberships, self.m)

    def objective(self, x):
        if self.memberships is None or self.centers is None:
            return np.infty
        distances = self.distances(x)
        return np.sum(self.fuzzifier(self.memberships) * distances)


class GKFCM(FCM):
    """Gives clusters ellipsoidal character.
    The Gustafson-Kessel algorithm redefines the distance measurement such that
    clusters may adopt ellipsoidal shapes. This is achieved through updates to
    a covariance matrix assigned to each cluster center.
    Examples
    --------
    Create a algorithm for probabilistic clustering with ellipsoidal clusters:
    >>> class ProbabilisticGustafsonKessel(GustafsonKesselMixin, Probabilistic):
    >>>     pass
    >>> pgk = ProbabilisticGustafsonKessel()
    >>> pgk.fit(x)
    """
    covariance = None

    def fit(self, x):
        """Optimizes cluster centers by restarting convergence several times.
        Extends the default behaviour by recalculating the covariance matrix
        with resultant memberships and centers.
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        """
        j_list = super(GKFCM, self).fit(x)
        self.covariance = self.calculate_covariance(x)
        return j_list

    def update(self, x):
        """Single update of the cluster algorithm.
        Extends the default behaviour by including a covariance calculation
        after updating the centers
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        """
        self.initialize(x)
        self.centers = self.calculate_centers(x)
        self.covariance = self.calculate_covariance(x)
        self.memberships = self.calculate_memberships(x)

    def distances(self, x):
        covariance = self.covariance if self.covariance is not None \
            else self.calculate_covariance(x)
        d = x - self.centers[:, np.newaxis]
        left_multiplier = \
            np.einsum('...ij,...jk', d, np.linalg.inv(covariance))
        return np.sum(left_multiplier * d, axis=2).T

    def calculate_covariance(self, x):
        """Calculates the covariance of the data `u` with cluster centers `v`.
        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.
        Returns
        -------
        :obj:`np.ndarray`
            (n_clusters, n_features, n_features)
            The covariance matrix of each cluster.
        """
        v = self.centers
        if v is None:
            return None
        q, p = v.shape
        if self.memberships is None:
            # If no memberships have been calculated assume n-spherical clusters
            return (np.eye(p)[..., np.newaxis] * np.ones((p, q))).T
        q, p = v.shape
        vector_difference = x - v[:, np.newaxis]
        fuzzy_memberships = self.fuzzifier(self.memberships)
        right_multiplier = \
            np.einsum('...i,...j->...ij', vector_difference, vector_difference)
        einstein_sum = \
            np.einsum('i...,...ijk', fuzzy_memberships, right_multiplier) / \
            np.sum(fuzzy_memberships, axis=0)[..., np.newaxis, np.newaxis]
        return np.nan_to_num(
            einstein_sum / (np.linalg.det(einstein_sum) ** (1 / q))[
                ..., np.newaxis, np.newaxis])
