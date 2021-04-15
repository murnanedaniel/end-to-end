import pykeops
from pykeops.torch import LazyTensor
import torch
import numpy as np
from time import time as tt


def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64

    start = tt()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    #     print(c)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        cl_indices = torch.argsort(cl)
        cl_sorted = cl[cl_indices]

        _, counts = torch.unique(cl_sorted, return_counts=True, sorted=True)
        cl_indices_split = cl_indices.split(tuple(counts))

        buckets = []
        for bucket in cl_indices_split:
            buckets.append(torch.combinations(bucket))

        buckets_cat = torch.cat(buckets).t()
        dist = torch.sum((x[buckets_cat[0]] - x[buckets_cat[1]]) ** 2, axis=-1)

        r = 0.4
        r_mask = dist < r ** 2

        edges = buckets_cat[:, r_mask]

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
    #         c.zero_()
    #         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

    #         # Divide by the number of points per cluster:
    #         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
    #         c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()

        #         print(cl_indices_split)
        print("Bucket size:", buckets_cat.shape)
        print("Edge size:", edges.shape)
        print("Average edge num:", edges.shape[1] / len(x))

        end = tt()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def main():
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64

    pykeops.clean_pykeops()  # just in case old build files are still present
    #     pykeops.test_torch_bindings()    # perform the compilation

    #     d = 2                           # dimension
    #     nb = 100000                      # database size
    #     np.random.seed(0)             # make reproducible
    #     xb = torch.rand(nb, d).type(dtype)

    #     M = 1000 if use_cuda else 100
    #     tmp = torch.linspace(0, 1, M).type(dtype)
    #     g2, g1 = torch.meshgrid(tmp, tmp)
    #     g = torch.cat((g1.contiguous().view(-1, 1), g2.contiguous().view(-1, 1)), dim=1)

    #     G_i = LazyTensor(g[:, None, :])  # (M**2, 1, 2)
    #     X_j = LazyTensor(xb[None, :, :])  # (1, N, 2)
    #     D_ij = ((G_i - X_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances

    #     start = tt()
    #     indKNN = D_ij.argKmin(10, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
    #     end = tt()

    #     print("nbs: {}".format(indKNN[0]))
    #     print("Time: {}s".format(end-start))

    #     start = tt()
    #     indKNN = D_ij.argKmin(10, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
    #     end = tt()

    #     print("nbs: {}".format(indKNN[0]))
    #     print("Time: {}s".format(end-start))

    N, D, num_nb = 100000, 8, 1000
    K = int(N / num_nb)

    x = torch.rand(N, D, dtype=dtype).to(0)

    cl, c = KMeans(x, K)

    cl, c = KMeans(x, K)

    print("Clusters:", cl)


if __name__ == "__main__":

    main()
