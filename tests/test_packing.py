import pytest
import torch

from bitlattice.packing import pack, unpack


@pytest.mark.parametrize("bits", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_bytes_match_bitwise_assembly(bits, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    codes = torch.randint(-64, 65, (17, 1024), dtype=dtype, device=device)
    codes[0].fill_(-1)
    codes[1].zero_()
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    expected = torch.zeros(17, 1024 // per_byte, dtype=torch.uint8, device=device)
    for offset in range(per_byte):
        part = (codes[:, offset::per_byte].long() & mask) << (offset * bits)
        expected.bitwise_or_(part.to(torch.uint8))
    actual = pack(codes, bits)
    assert actual.dtype == torch.uint8
    assert torch.equal(actual, expected)
    assert torch.equal(unpack(actual, bits, 1024), codes.int() & mask)
