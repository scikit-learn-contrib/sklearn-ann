from sklearn.neighbors import KNeighborsTransformer
from functools import partial


BallTreeTransformer = partial(KNeighborsTransformer, algorithm="ball_tree")
KDTreeTransformer = partial(KNeighborsTransformer, algorithm="kd_tree")
BruteTransformer = partial(KNeighborsTransformer, algorithm="brute")


__all__ = ["BallTreeTransformer", "KDTreeTransformer", "BruteTransformer"]
