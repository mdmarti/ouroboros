import pytest
import torch

from model.model import Ouroboros
from model.kernels import fullPolyModule


def z_of(model, x, dxdt, dt):
    """the (y, tau * ydot) input the model builds internally"""
    return torch.cat([x, dxdt * model.tau / dt], dim=-1)


# --------------------------------------------------------------------- construction


def test_attributes(model, dt):
    assert model.tau == dt
    assert model.kernel.tau == dt  # kernel is told the model timescale
    assert model.kernel.nTerms == 4
    assert len(model.names) == 4


def test_three_independent_encoders(model):
    ids = {id(model.omega_mamba), id(model.gamma_mamba), id(model.kernel_mamba)}
    assert len(ids) == 3
    # (A_log is deterministically initialised, so compare a randomly initialised weight)
    w = lambda m: m.layers[0].mixer.in_proj.weight
    assert not torch.equal(w(model.omega_mamba), w(model.gamma_mamba))
    assert not torch.equal(w(model.omega_mamba), w(model.kernel_mamba))


# ------------------------------------------------------------------------- forward


def test_forward_shapes_and_finiteness(model, batch, dt):
    x, dxdt = batch
    yhat, weights = model(x, dxdt, dt)
    assert yhat.shape == x.shape
    assert weights.shape == (2, 600, 5, 5)
    assert torch.isfinite(yhat).all() and torch.isfinite(weights).all()


def test_forward_does_not_modify_its_inputs(model, batch, dt):
    x, dxdt = batch
    x0, d0 = x.clone(), dxdt.clone()
    model(x, dxdt, dt)
    assert torch.equal(x, x0)
    assert torch.equal(dxdt, d0), "forward rescaled the caller's dxdt in place"


def test_forward_is_repeatable(model, batch, dt):
    x, dxdt = batch
    a, _ = model(x, dxdt, dt)
    b, _ = model(x, dxdt, dt)
    torch.testing.assert_close(a, b)


def test_forward_matches_the_oscillator_equation(model, batch, dt):
    # yhat = -omega^2 y - gamma (tau ydot) - g(y, tau ydot)
    x, dxdt = batch
    yhat, weights = model(x, dxdt, dt)
    omega, gamma, kernel, w, _ = model.get_funcs(x, dxdt, dt)
    z = z_of(model, x, dxdt, dt)
    expected = -(omega**2) * z[:, :, :1] - gamma * z[:, :, 1:] - kernel
    torch.testing.assert_close(yhat, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(weights, w)


def test_kernel_term_is_consistent_with_weights(model, batch, dt):
    x, dxdt = batch
    _, _, kernel, weights, _ = model.get_funcs(x, dxdt, dt)
    z = z_of(model, x, dxdt, dt)
    rebuilt = model.kernel.forward_given_weights(z, weights.clone())
    torch.testing.assert_close(kernel, rebuilt, atol=1e-6, rtol=1e-5)


def test_omega_is_nonnegative(model, batch, dt):
    x, dxdt = batch
    omega, *_ = model.get_funcs(x, dxdt, dt)
    assert (omega >= 0).all()


def test_tau_and_dt_only_enter_through_scaled_derivative(make_model, batch):
    # model sees tau * dxdt / dt, so (tau, dt) -> (2 tau, 2 dt) leaves its input unchanged
    x, dxdt = batch
    m1 = make_model(tau=1e-4)
    m2 = make_model(tau=2e-4)
    m2.load_state_dict(m1.state_dict())
    a, _ = m1(x, dxdt, 1e-4)
    b, _ = m2(x, dxdt, 2e-4)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_forward_smoothing_flag(model, batch, dt):
    x, dxdt = batch
    plain, _ = model(x, dxdt, dt, smoothing=False)
    smoothed, _ = model(x, dxdt, dt, smoothing=True)
    assert smoothed.shape == plain.shape
    assert not torch.allclose(plain, smoothed)


def test_batch_elements_are_independent(model, batch, dt):
    x, dxdt = batch
    both, _ = model(x, dxdt, dt)
    first, _ = model(x[:1], dxdt[:1], dt)
    torch.testing.assert_close(both[:1], first, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------- get_funcs


def test_get_funcs_outputs(model, batch, dt):
    x, dxdt = batch
    omega, gamma, kernel, weights, states = model.get_funcs(x, dxdt, dt)
    assert omega.shape == gamma.shape == kernel.shape == (2, 600, 1)
    assert weights.shape == (2, 600, 5, 5)
    assert states.shape == (2, 600, 6)  # omega, gamma, kernel controls (2 dims each)


def test_get_funcs_does_not_modify_inputs(model, batch, dt):
    x, dxdt = batch
    d0 = dxdt.clone()
    model.get_funcs(x, dxdt, dt)
    assert torch.equal(dxdt, d0)


def test_get_funcs_smoothing_is_causal_average(model, batch, dt):
    x, dxdt = batch
    raw_o, raw_g, *_ = model.get_funcs(x, dxdt, dt, smoothing=False)
    sm_o, sm_g, *_ = model.get_funcs(x, dxdt, dt, smoothing=True)
    n = int(round(model.smooth_len / dt))
    # mean of the trailing window of the raw function
    torch.testing.assert_close(
        sm_o[:, n + 5], raw_o[:, 6 : n + 6].mean(dim=1), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        sm_g[:, n + 5], raw_g[:, 6 : n + 6].mean(dim=1), atol=1e-6, rtol=1e-5
    )


# ------------------------------------------------------------------ funcs_by_step


@pytest.mark.parametrize("step_size", [7, 64, 300, 599, 600, 601, 5000])
def test_stepwise_matches_full_pass(model, batch, dt, step_size):
    # chunked processing carries encoder state across chunks, so it must agree
    # with one big pass; step sizes straddle the reversed/forward boundary (L = 600)
    x, dxdt = batch
    omega, gamma, kernel, weights, _ = model.get_funcs(x, dxdt, dt)
    z = z_of(model, x, dxdt, dt)
    yhat, o, g, k, w = model.funcs_by_step(z, dt, step_size=step_size)
    for full, step in [(omega, o), (gamma, g), (kernel, k), (weights, w)]:
        assert full.shape == step.shape
        torch.testing.assert_close(full, step, atol=1e-5, rtol=1e-4)
    full_yhat, _ = model(x, dxdt, dt)
    torch.testing.assert_close(full_yhat, yhat, atol=1e-5, rtol=1e-4)


def test_stepwise_smoothing_matches_full_pass(model, batch, dt):
    x, dxdt = batch
    omega, gamma, *_ = model.get_funcs(x, dxdt, dt, smoothing=True)
    z = z_of(model, x, dxdt, dt)
    _, o, g, *_ = model.funcs_by_step(z, dt, smoothing=True, step_size=128)
    torch.testing.assert_close(omega, o, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(gamma, g, atol=1e-5, rtol=1e-4)


def test_stepwise_uses_each_encoders_own_state(model, batch, dt):
    # regression test: gamma and the kernel used to be computed with the omega encoder
    x, dxdt = batch
    z = z_of(model, x, dxdt, dt)
    _, o, g, k, w = model.funcs_by_step(z, dt, step_size=100)
    assert not torch.allclose(o, g)
    assert w.abs().sum() > 0


def test_get_funcs_switches_to_stepwise_for_long_audio(model, batch, dt, capsys):
    x, dxdt = batch
    full = model.get_funcs(x, dxdt, dt)
    # max_len_t is in seconds: 0.01 s = 200 samples < 600
    long = model.get_funcs(x, dxdt, dt, max_len_t=0.01)
    assert "step by step" in capsys.readouterr().out
    for a, b in zip(full[:4], long[:4]):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    assert long[4] == []


def test_stepwise_does_not_modify_input(model, batch, dt):
    x, dxdt = batch
    z = z_of(model, x, dxdt, dt)
    z0 = z.clone()
    model.funcs_by_step(z, dt, step_size=100)
    assert torch.equal(z, z0)


# ---------------------------------------------------------------------- training


def test_gradients_reach_every_parameter(model, batch, dt):
    model.train()
    x, dxdt = batch
    yhat, weights = model(x, dxdt, dt)
    (yhat**2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    assert model.omega_net.weight.grad.abs().sum() > 0
    assert model.gamma_net.weight.grad.abs().sum() > 0
    assert model.kernel.weights.weight.grad.abs().sum() > 0


def test_a_few_steps_reduce_the_loss(make_model, dt):
    # sanity: the model can fit a damped oscillator's second derivative a little
    model = make_model(n_layers=1, d_state=4, expand_factor=2)
    model.train()
    t = torch.arange(400) * dt
    y = torch.sin(2 * torch.pi * 500 * t)[None, :, None].repeat(4, 1, 1)
    dy = torch.cos(2 * torch.pi * 500 * t)[None, :, None].repeat(4, 1, 1) * (
        2 * torch.pi * 500 * dt
    )
    d2y = -((2 * torch.pi * 500 * dt) ** 2) * y
    target = d2y / dt**2 * model.tau**2
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(25):
        opt.zero_grad()
        yhat, _ = model(y, dy, dt)
        loss = ((yhat - target) ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_integrate_is_a_noop_that_points_to_eval(model, capsys):
    assert model.integrate(None, None, None, None) is None
    assert "train/eval.py" in capsys.readouterr().out


def test_state_dict_roundtrip(make_model, batch, dt):
    a, b = make_model(), make_model()
    b.load_state_dict(a.state_dict())
    x, dxdt = batch
    ya, _ = a(x, dxdt, dt)
    yb, _ = b(x, dxdt, dt)
    torch.testing.assert_close(ya, yb)


def test_different_kernel_degrees_change_weight_shape(make_model, batch, dt):
    x, dxdt = batch
    for n in (2, 3, 6):
        _, w = make_model(n_terms=n)(x, dxdt, dt)
        assert w.shape[-2:] == (n + 1, n + 1)
