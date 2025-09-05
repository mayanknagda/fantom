"""\nMetrics for evaluating topic models (coherence, diversity, etc.).\n\nThis module is part of the `tomo` topic modeling library.\n"""

from ._coherence import return_coherence
from ._diversity import return_topic_diversity

__all__ = ["return_coherence", "return_topic_diversity"]
