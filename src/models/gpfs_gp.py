"""
Code for Gaussian processes using GPflow and GPflowSampling.
"""

from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float as floatx

from .simple_gp import SimpleGp
from .gpfs.models import PathwiseGPR
from .gp.gp_utils import kern_exp_quad
from ..utils.base import Base
from ..utils.misc_util import dict_to_namespace, suppress_stdout_stderr
from ..utils.domain_util import unif_random_sample_domain


class GpfsGp(SimpleGp):
    """
    GP model using GPFlowSampling.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'GpfsGp')
        self.params.n_bases = getattr(params, 'n_bases', 1000)
        self.params.n_dimx = getattr(params, 'n_dimx', 1)
        self.set_kernel(params)

    def set_kernel(self, params):
        """Set GPflow kernel."""
        self.params.kernel_str = getattr(params, 'kernel_str', 'rbf')

        ls = self.params.ls
        kernvar = self.params.alpha**2

        if self.params.kernel_str == 'rbf':
            gpf_kernel = kernels.SquaredExponential(variance=kernvar, lengthscales=ls)
            kernel = getattr(params, 'kernel', kern_exp_quad)
        elif self.params.kernel_str == 'matern52':
            gpf_kernel = kernels.Matern52(variance=kernvar, lengthscales=ls)
            raise Exception('Matern 52 kernel is not yet supported.')
        elif self.params.kernel_str == 'matern32':
            gpf_kernel = kernels.Matern32(variance=kernvar, lengthscales=ls)
            raise Exception('Matern 32 kernel is not yet supported.')

        self.params.gpf_kernel = gpf_kernel
        self.params.kernel = kernel

    def set_data(self, data):
        """Set self.data."""
        super().set_data(data)
        self.tf_data = Namespace()
        self.tf_data.x = tf.convert_to_tensor(np.array(self.data.x))
        self.tf_data.y = tf.convert_to_tensor(
            np.array(self.data.y).reshape(-1, 1)
        )
        self.set_model()

    def set_model(self):
        """Set GPFlowSampling as self.model."""
        self.params.model = PathwiseGPR(
            data=(self.tf_data.x, self.tf_data.y),
            kernel=self.params.gpf_kernel,
            noise_variance=self.params.sigma**2,
        )

    def initialize_function_sample_list(self, n_samp=1):
        """Initialize a list of n_samp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_samp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)

        Xinit = tf.random.uniform(
            [n_samp, self.params.n_dimx], minval=0.0, maxval=0.1, dtype=floatx()
        )
        Xvars = tf.Variable(Xinit)
        self.fsl_xvars = Xvars

    @tf.function
    def call_fsl_on_xvars(self, model, xvars, sample_axis=0):
        """Call fsl on fsl_xvars."""
        fvals =  model.predict_f_samples(Xnew=xvars, sample_axis=sample_axis)
        return fvals

    def call_function_sample_list(self, x_list):
        """Call a set of posterior function samples on respective x in x_list."""

        # Replace Nones in x_list with first non-None value
        x_list = self.replace_x_list_none(x_list)

        # Set fsl_xvars as x_list, call fsl, return y_list
        self.fsl_xvars.assign(x_list)
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_list = list(y_tf.numpy().reshape(-1))
        return y_list

    def replace_x_list_none(self, x_list):
        """Replace any Nones in x_list with first non-None value and return x_list."""

        # Set new_val as first non-None element of x_list
        new_val = next(x for x in x_list if x is not None)

        # Replace all Nones in x_list with new_val
        x_list_new = [new_val if x is None else x for x in x_list]

        return x_list_new


class MultiGpfsGp(Base):
    """
    Simple multi-output GP model using GPFlowSampling. To do this, this class duplicates
    the model in GpfsGp multiple times (and uses same kernel and other parameters in
    each duplication).
    """

    def __init__(self, params=None, data=None, verbose=True):
        super().__init__(params, verbose)
        self.set_data(data)
        self.set_gpfsgp_list()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, 'name', 'MultiGpfsGp')
        self.params.n_dimy = getattr(params, 'n_dimy', 1)
        self.params.gp_params = getattr(params, 'gp_params', {})

    def set_data(self, data):
        """Set self.data."""
        if data is None:
            self.data = Namespace(x=[], y=[])
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)

    def set_gpfsgp_list(self):
        """Set self.gpfsgp_list by instantiating a list of GpfsGp objects."""
        data_list = self.get_data_list(self.data)

        # NOTE: GpfsGp verbose set to False (though MultiGpfsGp may be verbose)
        self.gpfsgp_list = [
            GpfsGp(self.params.gp_params, dat, False) for dat in data_list
        ]

    def initialize_function_sample_list(self, n_samp=1):
        """
        Initialize a list of n_samp function samples, for each GP in self.gpfsgp_list.
        """
        for gpfsgp in self.gpfsgp_list:
            gpfsgp.initialize_function_sample_list(n_samp)

    def call_function_sample_list(self, x_list):
        """
        Call a set of posterior function samples on respective x in x_list, for each GP
        in self.gpfsgp_list.
        """
        y_list_list = [
            gpfsgp.call_function_sample_list(x_list) for gpfsgp in self.gpfsgp_list
        ]

        # y_list is a list, where each element is a list representing a multidim y
        y_list = [list(x) for x in zip(*y_list_list)]
        return y_list

    def get_post_mu_cov(self, x_list, full_cov=False):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""
        mu_list, cov_list = self.gp_post_wrapper(x_list, self.data, full_cov)
        return mu_list, cov_list

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""

        data_list = self.get_data_list(data)
        mu_list = []
        cov_list = []

        for gpfsgp, data_single in zip(self.gpfsgp_list, data_list):
            # Call usual 1d gpfsgp gp_post_wrapper
            mu, cov = gpfsgp.gp_post_wrapper(x_list, data_single, full_cov)
            mu_list.append(mu)
            cov_list.append(cov)

        return mu_list, cov_list

    def get_data_list(self, data):
        """
        Return list of Namespaces, where each is a version of data containing only one
        of the dimensions of data.y (and the full data.x).
        """
        data_list = []
        for j in range(self.params.n_dimy):
            data_list.append(Namespace(x=data.x, y=[yi[j] for yi in data.y]))

        return data_list
