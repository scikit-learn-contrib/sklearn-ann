from functools import partial

from sklearn.neighbors import KNeighborsTransformer

BallTreeTransformer = partial(KNeighborsTransformer, algorithm="ball_tree")
KDTreeTransformer = partial(KNeighborsTransformer, algorithm="kd_tree")
BruteTransformer = partial(KNeighborsTransformer, algorithm="brute")


__all__ = ["BallTreeTransformer", "KDTreeTransformer", "BruteTransformer"]
