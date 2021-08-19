import numpy as np

import torch
import torch.distributed as dist

class DistDataError(Exception):
    """Defines an empty exception to throw when some other rank hit a real exception."""
    pass

class DistData(object):
    def __init__(self, backend='gloo'):
        assert backend in ['gloo', 'mpi'], f"torch.distributed backend '{backend}' is not supported, valid options are 'gloo' or 'mpi'"

        dist.init_process_group(backend, init_method="env://")

        # lookup our process rank and the group size
        self.rank = dist.get_rank()
        self.numranks = dist.get_world_size()

    def allassert(self, cond, msg):
        """Check that cond is True on all ranks, assert with msg everywhere if not."""
        alltrue = self.alltrue(cond)
        assert alltrue, msg

    def allraise_if(self, err):
        """Raise exception if err is not None on any rank."""
        alltrue = self.alltrue(err is None)
        if not alltrue:
            # At least one rank raised an exception.
            # Re-raise the actual exception if this rank threw one.
            if err is not None:
                raise err

            # TODO: is there a better exception to use here?
            # On other ranks, raise an "empty" exception to indicate
            # that we're only failing because someone else did.
            raise DistDataError

    def barrier(self):
        """Globally synchronize all processes"""
        dist.barrier()

    def bcast(self, val, root):
        """Broadcast a scalar value from root to all ranks"""
        vals = [val]
        dist.broadcast_object_list(vals, src=root)
        return vals[0]

    def bcast_list(self, vals, root=0):
        """Broadcast list of vals from root to all ranks, returns newly allocated list"""
        # broadcast length of vals list
        length = [len(vals)]
        dist.broadcast_object_list(length, src=root)

        # allocate a tensor of appropriate size
        # initialize tensor with list values on root
        if self.rank == root:
            tvals = torch.tensor(vals, dtype=torch.int64)
        else:
            tvals = torch.zeros(length[0], dtype=torch.int64)

        # broadcast tensor from root, and return as a new list
        dist.broadcast(tvals, src=root)
        return tvals.tolist()

    def scatterv_(self, invals, counts, outval, root=0):
        """Scatter int64 values from invals according to counts array, receive values in outval"""

        self.allassert(len(counts) == self.numranks,
            f"Length of counts list {len(counts)} does not match number of ranks {self.numranks}")

        self.allassert(outval.shape == (counts[self.rank],),
            f"Rank {self.rank}: output buffer is of shape {outval.shape}, expected {(counts[self.rank],)}")

        self.allassert(outval.dtype == np.int64,
            f"Requires outval to be of type numpy.int64")

        scatterlist = None
        if self.rank == root:
            scatterlist = list(torch.split(torch.from_numpy(invals), counts))
        outtensor = torch.from_numpy(outval)
        dist.scatter(outtensor, scatterlist, src=root)

    def alltrue(self, val):
        """Returns True if all procs input True, False otherwise"""
        # torch.dist does not support reductions with bool types
        # so we cast to int and cast the result back to bool
        tensor = torch.tensor([int(val)], dtype=torch.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.BAND)
        return bool(tensor[0])

    def sum(self, val):
        """Compute sum of a scalar val, and return total on all ranks."""
        tensor = torch.tensor([val])
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor[0]

    def exscan(self, val):
        """Compute prefix sum (exclusive scan) of scalar val, and return offset of each rank."""
        # torch.distributed doesn't have a scan, so fallback to allreduce
        tensor = torch.zeros(self.numranks, dtype=torch.int64)
        tensor[self.rank:] = val
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor[self.rank]) - val

    def min(self, val):
        """Return minimum of scalar val to all ranks."""
        tensor = torch.tensor([val])
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return tensor[0]

    def minrank(self, cond):
        """Find first rank whose condition is True, return that rank if any, None otherwise."""
        minrank = self.numranks
        if cond:
            minrank = self.rank
        minrank = self.min(minrank)

        if minrank < self.numranks:
            return minrank
        return None

    def bcast_first(self, val):
        """Broadcast val from first rank where it is not None, return val if any, None otherwise"""
        # Find the first rank with a valid value.
        minrank = self.minrank(val is not None)

        # If there is no rank with a valid value, return None
        if minrank is None:
            return None

        # Otherwise broadcast the value from the first valid rank.
        val = self.bcast(val, root=minrank)
        return val

    def all_sum_(self, vals):
        """Sums values in numpy array vals element-wise and update vals in place with final result on all ranks"""
        tensor = torch.from_numpy(vals)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def open(self, filename):
        """Create, truncate, and open a file shared by all ranks."""

        # Don't truncate existing file until all ranks reach this point
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 creates and truncates file.
        if self.rank == 0:
            try:
                f = open(filename, 'wb')
            except Exception as e:
                err = e

        # Verify that rank 0 created the file
        self.allraise_if(err)

        # Wait for rank 0 to open (and truncate) file,
        # then have all ranks open file for writing.
        if self.rank != 0:
            try:
                f = open(filename, 'r+b')
            except Exception as e:
                err = e

        # Verify that all ranks successfully opened the file
        self.allraise_if(err)

        return f
