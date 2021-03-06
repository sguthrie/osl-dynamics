"""Base class for simulations.

"""

from vrad.data.manipulation import standardize
from vrad.utils import plotting


class Simulation:
    """Simulation base class.

    Parameters
    ----------
    n_samples : int
        Number of time points to generate.
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.time_series = None

    def __array__(self):
        return self.time_series

    def __iter__(self):
        return iter([self.time_series])

    def __getattr__(self, attr):
        if attr == "time_series":
            raise NameError("time_series has not yet been created.")
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self.time_series, attr)

    def __len__(self):
        return 1

    def standardize(self):
        self.time_series = standardize(self.time_series, axis=0)

    def plot_data(self, n_points: int = 1000, filename: str = None):
        """Method for plotting simulated data.

        Parameters
        ----------
        n_points : int
            Number of time points to plot.
        filename : str
            Filename to save plot to.
        """
        n_points = min(n_points, self.n_samples)
        plotting.plot_time_series(
            self.time_series, n_samples=n_points, filename=filename
        )
