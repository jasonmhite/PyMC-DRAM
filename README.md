PyMC-DRAM
=========

This package implements a Delayed Rejection Adaptive Metropolis sampler
in PyMC 2. 

Installing
==========

* Install PyMC 2
* Clone this repository
* `python setup.py install` or `python setup.py develop` (you may need `sudo` too)

Use
===

```python
import pymc as P
from pymc_dram import DRAM

NS = ...

# do initialization stuff
...

# construct model
def model_factory():
    ...
    return locals()

mvars = model_factory()
M = P.MCMC(mvars)

M.use_step_method(
    DRAM,
    [x for (_, x) in mvars.items()],
)

M.sample(NS, burn=NS / 2.)

# analyze chains
...
```

For a more complete example, see [`test_model.py`](test/test_model.py).
This recreates the results of Example 8.2 in *Uncertainty Quantification:
Theory, Implementation, and Applications* by Ralph Smith.

To get a list of parameters you can pass to the sampler, see the docstring
(`help(DRAM)` in the interpreter or browse the [source](pymc_dram/dram.py)).
The default parameters should be fine for most users.

Notes
=====

This only computes one delayed rejection step. It could be extended to
add more but it seems most implementations don't bother and it adds
significant complexity for marginal benefits.

License
=======

This software shares the same license as PyMC, the Academic Free License 3.0,
see [LICENSE](LICENSE).

Credit
======
This code is adapted from an initial implementation by Kyle Huston (@khuston),
see the [credits file](CREDITS.md). 
