from __future__ import division

import numpy as np
import pymc

from pymc.six import print_

class DRAM(pymc.AdaptiveMetropolis):
 
    """
    The DelayedRejectionAdaptativeMetropolis (DRAM) sampling algorithm works like
    AdaptiveMetropolis, except that if a bold initial jump proposal is rejected, a more
    conservative jump proposal will be tried. Although the chain is non-Markovian, as 
    with AdaptiveMetropolis, it too has correct ergodic properties. See (Haario et al.,
    2006) for details.
    :Parameters:
      - stochastic : PyMC objects
          Stochastic objects to be handled by the AM algorith,
      - cov : array
          Initial guess for the covariance matrix C. If it is None, the
          covariance will be estimated using the scales dictionary if provided,
          the existing trace if available, or the current stochastics value.
          It is suggested to provide a sensible guess for the covariance, and
          not rely on the automatic assignment from stochastics value.
      - delay : int
          Number of steps before the empirical covariance is computed. If greedy
          is True, the algorithm waits for delay *accepted* steps before computing
          the covariance.
      - interval : int
          Interval between covariance updates. Higher dimensional spaces require
          more samples to obtain reliable estimates for the covariance updates.
      - greedy : bool
          If True, only the accepted jumps are tallied in the internal trace
          until delay is reached. This is useful to make sure that the empirical
          covariance has a sensible structure.
      - shrink_if_necessary : bool
          If True, the acceptance rate is checked when the step method tunes. If
          the acceptance rate is small, the proposal covariance is shrunk according
          to the following rule:
          if acc_rate < .001:
              self.C *= .01
          elif acc_rate < .01:
              self.C *= .25
      - scales : dict
          Dictionary containing the scale for each stochastic keyed by name.
          If cov is None, those scales are used to define an initial covariance
          matrix. If neither cov nor scale is given, the initial covariance is
          guessed from the trace (it if exists) or the objects value, alt
      - verbose : int
          Controls the verbosity level.
    :Notes:
    Use the methods: `cov_from_scales`, `cov_from_trace` and `cov_from_values` for
    more control on the creation of an initial covariance matrix. A lot of problems
    can be avoided with a good initial covariance and long enough intervals between
    covariance updates. That is, do not compensate for a bad covariance guess by
    reducing the interval between updates thinking the covariance matrix will
    converge more rapidly.
    :Reference:
      Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm,
          Bernouilli, vol. 7 (2), pp. 223-242, 2001.
      Haario 2006.
    """

    def __init__(
        self,
        stochastic,
        cov=None,
        delay=1000,
        interval=200,
        greedy=True,
        drscale = 0.1,
        shrink_if_necessary=False,
        scales=None,
        verbose=-1, 
        tally=False,
    ):
        # Verbosity flag
        self.verbose = verbose

        self.accepted = 0
        self.rejected = 0 # Just a dummy variable for compatibility with the superclass
        self.rejected_then_accepted = 0
        self.rejected_twice = 0

        if not np.iterable(stochastic) or isinstance(stochastic, pymc.Variable):
            stochastic = [stochastic]

        # Initialize superclass
        pymc.StepMethod.__init__(self, stochastic, verbose, tally)

        self._id = 'DelayedRejectionAdaptiveMetropolis_' + '_'.join(
            [p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += [
            'accepted', 'rejected_then_accepted','rejected_twice', '_trace_count', '_current_iter', 'C', 'proposal_sd',
            '_proposal_deviate', '_trace', 'shrink_if_necessary']
        self._tuning_info = ['C']

        self.proposal_sd = None
        self.shrink_if_necessary = shrink_if_necessary

        # Number of successful steps before the empirical covariance is
        # computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy
        # Scale for second attempt
        self.drscale = drscale

        # Initialization methods
        self.check_type()
        self.dimension()

        # Set the initial covariance using cov, or the following fallback mechanisms:
        # 1. If scales is provided, use it.
        # 2. If a trace is present, compute the covariance matrix empirically from it.
        # 3. Use the stochastics value as a guess of the variance.
        if cov is not None:
            self.C = cov
        elif scales:
            self.C = self.cov_from_scales(scales)
        else:
            try:
                self.C = self.cov_from_trace()
            except AttributeError:
                self.C = self.cov_from_value(100.)

        self.updateproposal_sd()

        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy
        # sampling can be done during warm-up period.
        self._trace_count = 0
        self._current_iter = 0

        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []

        if self.verbose >= 2:
            print_("Initialization...")
            print_('Dimension: ', self.dim)
            print_("C_0: ", self.C)
            print_("Sigma: ", self.proposal_sd)

    def propose_first(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.
        The proposal jumps are drawn from a multivariate normal distribution.
        """

        arrayjump = np.dot(
            self.proposal_sd,
            np.random.normal(size=self.proposal_sd.shape[0]),
        )
        # save in case needed for calculating second proposal probability
        self.arrayjump1 = arrayjump.copy() # is a copy needed?

        if self.verbose > 2: print_('First jump:', arrayjump)

        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]].squeeze()

            if np.iterable(stochastic.value):
                jump = np.reshape(
                    arrayjump[self._slices[stochastic]],
                    np.shape(stochastic.value),
                )

            if self.isdiscrete[stochastic]:
                jump = round_array(jump)

            stochastic.value = stochastic.value + jump

    def propose_second(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.
        The proposal jumps are drawn from a multivariate normal distribution.
        """

        arrayjump = self.drscale * np.dot(
            self.proposal_sd,
            np.random.normal(size=self.proposal_sd.shape[0]),
        )

        if self.verbose > 2: print_('Second jump:', arrayjump)

        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]].squeeze()

            if np.iterable(stochastic.value):
                jump = np.reshape(
                    arrayjump[self._slices[stochastic]],
                    np.shape(stochastic.value),
                )

            if self.isdiscrete[stochastic]:
                jump = round_array(jump)

            stochastic.value = stochastic.value + jump

        arrayjump1 = self.arrayjump1
        arrayjump2 = arrayjump

        self.q = np.exp(
            -0.5 * (
                np.linalg.norm(np.dot(arrayjump2 - arrayjump1, self.proposal_sd_inv), ord=2) ** 2 \
                - np.linalg.norm(np.dot(-arrayjump1, self.proposal_sd_inv), ord=2) ** 2
            )
        )

    def step(self):
        """
        Perform a Metropolis step.
        Stochastic parameters are block-updated using a multivariate normal
        distribution whose covariance is updated every self.interval once
        self.delay steps have been performed.
        The AM instance keeps a local copy of the stochastic parameter's trace.
        This trace is used to computed the empirical covariance, and is
        completely independent from the Database backend.
        If self.greedy is True and the number of iterations is smaller than
        self.delay, only accepted jumps are stored in the internal
        trace to avoid computing singular covariance matrices.
        """

        # Probability and likelihood for stochastic's current value:
        # PROBLEM: I am using logp plus loglike everywhere... and I shouldn't be!! TODO
        logp = self.logp_plus_loglike

        if self.verbose > 1:
            print_('Current value: ', self.stoch2array())
            print_('Current likelihood: ', logp)

        # Sample a candidate value
        self.propose_first()

        # Metropolis acception/rejection test
        accept = False

        try:
            # Probability and likelihood for stochastic's 1st proposed value:
            self.logp_p1 = self.logp_plus_loglike
            logp_p1 = float(self.logp_p1)
            self.logalpha01 = min(0, logp_p1 - logp)
            logalpha01 = self.logalpha01

            if self.verbose > 2:
                print_('First proposed value: ', self.stoch2array())
                print_('First proposed likelihood: ', logp_p1)

            if np.log(np.random.random()) < logalpha01:
                accept = True
                self.accepted += 1

                if self.verbose > 2:
                    print_('Accepted')
                logp_p = logp_p1
            else:
                if self.verbose > 2:
                    print_('Delaying rejection...')

                for stochastic in self.stochastics:
                    stochastic.revert()

                self.propose_second()

                try:
                    # Probability and likelihood for stochastic's 2nd proposed value:
                    # CHECK THAT THIS IS RECALCULATED WITH PROPOSE_SECODN
                    # CHECK THAT logp_p1 iS NOT CHANGED WHEN THIS IS RECALCD
                    logp_p2 = self.logp_plus_loglike
                    logalpha21 = min(0, logp_p1 - logp_p2)
                    l = logp_p2 - logp
                    q = self.q

                    logalpha_02 = np.log(
                        l * q * (1 - np.exp(logalpha21)) \
                        / (1 - np.exp(logalpha01))
                    )

                    if self.verbose > 2:
                        print_('Second proposed value: ', self.stoch2array())
                        print_('Second proposed likelihood: ', logp_p2)

                    if np.log(np.random.random()) < logalpha_02:
                        accept = True
                        self.rejected_then_accepted += 1
                        logp_p = logp_p2
                        if self.verbose > 2:
                            print_('Accepted after one rejection')

                    else:
                        self.rejected_twice += 1
                        logp_p = None

                        if self.verbose > 2:
                            print_('Rejected twice')

                except pymc.ZeroProbability:
                    self.rejected_twice += 1
                    logp_p = None

                    if self.verbose > 2:
                        print_('Rejected twice')

        except pymc.ZeroProbability:
            if self.verbose > 2:
                print_('Delaying rejection...')

            for stochastic in self.stochastics:
                stochastic.revert()

            self.propose_second()

            try:
                # Probability and likelihood for stochastic's proposed value:
                logp_p2 = self.logp_plus_loglike
                logp_p1 = -np.inf
                logalpha01 = -np.inf
                logalpha21 = min(0, logp_p1 - logp_p2)
                l = np.exp(logp_p2 - logp)
                q = self.q

                logalpha_02 = np.log(
                    l * q * (1 - np.exp(logalpha21)) \
                    / (1 - np.exp(logalpha01))
                )

                if self.verbose > 2:
                    print_('Second proposed value: ', self.stoch2array())
                    print_('Second proposed likelihood: ', logp_p2)
                
                if np.log(np.random.random()) < logalpha_02:
                    accept = True
                    self.rejected_then_accepted += 1
                    logp_p = logp_p2

                    if self.verbose > 2:
                        print_('Accepted after one rejection with ZeroProbability')
                else:
                    self.rejected_twice += 1
                    logp_p = None

                    if self.verbose > 2:
                        print_('Rejected twice')

            except pymc.ZeroProbability:
                self.rejected_twice += 1
                logp_p = None

                if self.verbose > 2:
                    print_('Rejected twice with ZeroProbability Error.')
        #print_('\n\nRejected then accepted number of times: ',self.rejected_then_accepted)
        #print_('Rejected twice number of times: ',self.rejected_twice)

        if (not self._current_iter % self.interval) and self.verbose > 1:
            print_("Step ", self._current_iter)
            print_("\tLogprobability (current, proposed): ", logp, logp_p)

            for stochastic in self.stochastics:
                print_(
                    "\t",
                    stochastic.__name__,
                    stochastic.last_value,
                    stochastic.value,
                )

            if accept:
                print_("\tAccepted\t*******\n")

            else:
                print_("\tRejected\n")

            print_(
                "\tAcceptance ratio: ",
                (self.accepted + self.rejected_then_accepted) / (
                    self.accepted + 2.*self.rejected_then_accepted + 2.*self.rejected_twice
                )
            )

        if self._current_iter == self.delay:
            self.greedy = False

        if not accept:
            self.reject()

        if accept or not self.greedy:
            self.internal_tally()

        if self._current_iter > self.delay and self._current_iter % self.interval == 0:
            self.update_cov()

        self._current_iter += 1

    def update_cov(self):
        return super(DRAM, self).update_cov()

        self.rejected_then_accepted = 0.
        self.rejected_twice = 0. 

    def covariance_adjustment(self, f=0.9):
        self.proposal_sd *= f
        self.proposal_sd_inv = np.linalg.inv(self.proposal_sd)

    def updateproposal_sd(self):
        self.proposal_sd = np.linalg.cholesky(self.C)
        self.proposal_sd_inv = np.linalg.inv(self.proposal_sd)
