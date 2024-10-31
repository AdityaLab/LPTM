import pytest
import torch

from lptm.segment.scoring import (
    ScoringModuleAddn,
    ScoringModuleEnergy,
    ScoringModuleMult,
)


@pytest.mark.parametrize("embed_size", [15])
@pytest.mark.parametrize("hidden_size", [10, 15])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("seq_len", [5, 1, 15])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("add_mask", [True, False])
def test_addn(
    embed_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    add_mask: bool,
) -> None:
    scoring = ScoringModuleAddn(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(batch_size, seq_len, embed_size).to(device)
    if add_mask:
        mask = torch.zeros(batch_size, seq_len).bool().to(device)
        mask[:, -1] = True
    else:
        mask = None
    scores = scoring(time_embeds, mask)
    assert scores.size() == (batch_size, seq_len, seq_len)


@pytest.mark.parametrize("embed_size", [15])
@pytest.mark.parametrize("hidden_size", [10, 15])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("seq_len", [5, 1, 15])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("add_mask", [True, False])
def test_mult(
    embed_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    add_mask: bool,
) -> None:
    scoring = ScoringModuleMult(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(batch_size, seq_len, embed_size).to(device)
    if add_mask:
        mask = torch.zeros(batch_size, seq_len).bool().to(device)
        mask[:, -1] = True
    else:
        mask = None
    scores = scoring(time_embeds, mask)
    assert scores.size() == (batch_size, seq_len, seq_len)


@pytest.mark.parametrize("embed_size", [15])
@pytest.mark.parametrize("hidden_size", [10, 15])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("seq_len", [5, 1, 15])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("add_mask", [True, False])
def test_energy(
    embed_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    add_mask: bool,
) -> None:
    scoring = ScoringModuleEnergy(embed_size, hidden_size).to(device)
    time_embeds = torch.randn(batch_size, seq_len, embed_size).to(device)
    if add_mask:
        mask = torch.zeros(batch_size, seq_len).bool().to(device)
        mask[:, -1] = True
    else:
        mask = None
    scores = scoring(time_embeds, mask)
    assert scores.size() == (batch_size, seq_len, seq_len)
