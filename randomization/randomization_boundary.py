from dataclasses import dataclass

from randomization.randomization_bound import RandomizationBound
from randomization.randomization_parameter import RandomizationParameter


@dataclass
class RandomizationBoundary:
    """
    Describes the boundary sampled during Auto DR by the parameter and bound .
    """

    parameter: RandomizationParameter
    bound: RandomizationBound
