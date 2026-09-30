from .discrete import DiscreteAct, discrete
from .loss import causal_lm_loss, linear_cross_entropy

__all__ = ["DiscreteAct", "discrete", "linear_cross_entropy", "causal_lm_loss"]
