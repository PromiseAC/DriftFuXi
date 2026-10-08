"""Opt-in CPU reference for the THREE FBGEMM operators used by this smoke test.

Not a FBGEMM implementation/performance replacement. No CUDA kernels are registered.
CompositeImplicitAutograd lets native indexing/padding provide autograd.
"""
import torch

_LIBRARIES = []

def install():
    for name in ('asynchronous_complete_cumsum', 'dense_to_jagged', 'jagged_to_padded_dense'):
        if hasattr(torch.ops.fbgemm, name):
            raise RuntimeError('Refusing to replace existing FBGEMM operators')
    lib = torch.library.Library('fbgemm', 'FRAGMENT')
    lib.define('asynchronous_complete_cumsum(Tensor lengths) -> Tensor')
    lib.define('dense_to_jagged(Tensor dense, Tensor[] offsets, int? total_L=None) -> (Tensor, Tensor[])')
    lib.define('jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value=0.) -> Tensor')

    def cumsum(lengths):
        assert lengths.device.type == 'cpu' and (lengths >= 0).all()
        return torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

    def dense_to_jagged(dense, offsets, total_L=None):
        assert dense.device.type == 'cpu' and len(offsets) == 1
        lengths = offsets[0].diff()
        assert lengths.numel() == dense.shape[0] and (lengths <= dense.shape[1]).all()
        mask = torch.arange(dense.shape[1])[None, :] < lengths[:, None]
        return dense[mask], offsets

    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.):
        assert values.device.type == 'cpu' and len(offsets) == len(max_lengths) == 1
        off = offsets[0].tolist()
        assert off[0] == 0 and off[-1] == len(values)
        n = max_lengths[0]
        rows = []
        for a, b in zip(off, off[1:]):
            row = values[a:min(b, a+n)]
            rows.append(torch.cat([row, values.new_full((n-len(row), *values.shape[1:]), padding_value)]))
        return torch.stack(rows)

    for name, fn in [('asynchronous_complete_cumsum', cumsum), ('dense_to_jagged', dense_to_jagged), ('jagged_to_padded_dense', jagged_to_padded_dense)]:
        lib.impl(name, fn, 'CompositeImplicitAutograd')
    _LIBRARIES.append(lib)


def check():
    """Check variable/zero lengths, padding, truncation and gradients by hand."""
    x = torch.arange(24., dtype=torch.double).reshape(3,4,2).requires_grad_()
    off = torch.ops.fbgemm.asynchronous_complete_cumsum(torch.tensor([2,0,3]))
    j = torch.ops.fbgemm.dense_to_jagged(x, [off])[0]
    assert torch.equal(j, torch.cat([x[0,:2], x[2,:3]]))
    y = torch.ops.fbgemm.jagged_to_padded_dense(j, [off], [4])
    mask = torch.arange(4)[None,:] < torch.tensor([2,0,3])[:,None]
    assert torch.equal(y, x * mask[:,:,None])
    y.sum().backward()
    assert torch.equal(x.grad, mask[:,:,None].expand_as(x).double())
    z = torch.ops.fbgemm.jagged_to_padded_dense(j, [off], [1], -1.)
    assert torch.equal(z[1], torch.full((1,2), -1., dtype=torch.double))
    assert torch.equal(z[2], x.detach()[2,:1])
    return {'roundtrip': True, 'zero_length': True, 'truncation': True, 'gradient': True}
