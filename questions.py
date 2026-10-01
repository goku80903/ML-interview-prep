"""
Question bank for the interview practice app.

Each concept has an optional `numpy` and/or `torch` entry. Each entry has:
  prompt   - shown to the user
  stub     - starting code (raise NotImplementedError)
  solution - reference solution
  tests    - python source defining `_t1()`, `_t2()`, ... and calling
             `_check("name", _tN)` for each. Appended after the student's
             code and a small harness, then executed as a real subprocess.

`numpy: None` (only used for count_params) means there's no meaningful
NumPy analogue for that concept.
"""

CONCEPTS = [
    {
        "id": "softmax",
        "title": "Softmax",
        "category": "Activations & Losses",
        "numpy": {
            "prompt": "Numerically-stable softmax along a given axis. Must not overflow/NaN on large inputs.",
            "stub": '''def softmax(x, axis=-1):
    # TODO: implement a numerically-stable softmax
    raise NotImplementedError''',
            "solution": '''def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)''',
            "tests": '''
def _t1():
    x = np.array([[1.0,2.0,3.0],[1000.0,1000.0,1000.0]])
    out = softmax(x, axis=1)
    assert out.shape == x.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(out >= 0)
_check("shape correct & rows sum to 1", _t1)

def _t2():
    x = np.array([[1000.0,1000.0,1000.0]])
    assert not np.any(np.isnan(softmax(x, axis=1))), "overflowed"
_check("numerically stable on large inputs", _t2)

def _t3():
    x = np.array([1.0,2.0,3.0])
    out = softmax(x, axis=0)
    expected = np.exp(x - x.max()); expected /= expected.sum()
    assert np.allclose(out, expected, atol=1e-6)
_check("matches expected values", _t3)
''',
        },
        "torch": {
            "prompt": "Same idea with torch tensors: numerically-stable softmax along a given dim. No torch.softmax shortcut.",
            "stub": '''def softmax(x, dim=-1):
    # TODO: implement a numerically-stable softmax
    raise NotImplementedError''',
            "solution": '''def softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)''',
            "tests": '''
def _t1():
    x = torch.tensor([[1.0,2.0,3.0],[1000.0,1000.0,1000.0]])
    out = softmax(x, dim=1)
    assert out.shape == x.shape
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(out >= 0)
_check("shape correct & rows sum to 1", _t1)

def _t2():
    x = torch.tensor([[1000.0,1000.0,1000.0]])
    assert not torch.any(torch.isnan(softmax(x, dim=1))), "overflowed"
_check("numerically stable on large inputs", _t2)

def _t3():
    x = torch.tensor([1.0,2.0,3.0])
    out = softmax(x, dim=0)
    expected = torch.exp(x - x.max()); expected = expected / expected.sum()
    assert torch.allclose(out, expected, atol=1e-6)
_check("matches expected values", _t3)
''',
        },
    },
    {
        "id": "sigmoid",
        "title": "Sigmoid & derivative",
        "category": "Activations & Losses",
        "numpy": {
            "prompt": "sigmoid(x) = 1/(1+exp(-x)), stable for large |x|. sigmoid_derivative(x) = sigmoid(x)*(1-sigmoid(x)).",
            "stub": '''def sigmoid(x):
    # TODO: numerically-stable sigmoid
    raise NotImplementedError

def sigmoid_derivative(x):
    # TODO: derivative of sigmoid at x
    raise NotImplementedError''',
            "solution": '''def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)''',
            "tests": '''
def _t1():
    x = np.array([-1000.0, 0.0, 1000.0])
    out = sigmoid(x)
    assert np.allclose(out, [0.0, 0.5, 1.0], atol=1e-6)
    assert not np.any(np.isnan(out))
_check("stable at extreme values", _t1)

def _t2():
    x = np.array([0.0, 2.0, -3.5])
    d = sigmoid_derivative(x)
    s = sigmoid(x)
    assert np.allclose(d, s * (1 - s), atol=1e-6)
_check("derivative matches s*(1-s)", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. No torch.sigmoid shortcut.",
            "stub": '''def sigmoid(x):
    # TODO
    raise NotImplementedError

def sigmoid_derivative(x):
    # TODO
    raise NotImplementedError''',
            "solution": '''def sigmoid(x):
    return torch.where(x >= 0, 1/(1+torch.exp(-x)), torch.exp(x)/(1+torch.exp(x)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)''',
            "tests": '''
def _t1():
    x = torch.tensor([-1000.0, 0.0, 1000.0])
    out = sigmoid(x)
    assert torch.allclose(out, torch.tensor([0.0,0.5,1.0]), atol=1e-6)
_check("stable at extreme values", _t1)

def _t2():
    x = torch.tensor([0.0, 2.0, -3.5])
    d = sigmoid_derivative(x)
    s = sigmoid(x)
    assert torch.allclose(d, s * (1 - s), atol=1e-6)
_check("derivative matches s*(1-s)", _t2)
''',
        },
    },
    {
        "id": "cross_entropy",
        "title": "Cross-entropy loss",
        "category": "Activations & Losses",
        "numpy": {
            "prompt": "Mean cross-entropy from raw logits (N,C) and integer labels (N,). Use the log-sum-exp trick.",
            "stub": '''def cross_entropy_loss(logits, labels):
    # TODO
    raise NotImplementedError''',
            "solution": '''def cross_entropy_loss(logits, labels):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    correct_logit = shifted[np.arange(len(labels)), labels]
    return float(np.mean(log_sum_exp - correct_logit))''',
            "tests": '''
def _t1():
    logits = np.array([[2.0,1.0,0.1],[0.1,0.2,3.0]])
    labels = np.array([0,2])
    loss = cross_entropy_loss(logits, labels)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    ref = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    assert np.isclose(loss, ref, atol=1e-6)
_check("matches independently-computed reference", _t1)

def _t2():
    loss = cross_entropy_loss(np.array([[1000.0,1.0,0.0]]), np.array([0]))
    assert np.isfinite(loss)
_check("stable on huge logits", _t2)
''',
        },
        "torch": {
            "prompt": "Same, but differentiable (usable in a training loop). No F.cross_entropy / nn.CrossEntropyLoss.",
            "stub": '''def cross_entropy_manual(logits, labels):
    # TODO
    raise NotImplementedError''',
            "solution": '''def cross_entropy_manual(logits, labels):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1))
    correct_logit = shifted[torch.arange(logits.size(0)), labels]
    return (log_sum_exp - correct_logit).mean()''',
            "tests": '''
def _t1():
    logits = torch.tensor([[2.0,1.0,0.1],[0.1,0.2,3.0]], requires_grad=True)
    labels = torch.tensor([0,2])
    loss = cross_entropy_manual(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits.detach(), labels)
    assert torch.isclose(loss.detach(), ref, atol=1e-5)
_check("matches torch's own cross_entropy", _t1)

def _t2():
    logits = torch.tensor([[2.0,1.0,0.1],[0.1,0.2,3.0]], requires_grad=True)
    labels = torch.tensor([0,2])
    loss = cross_entropy_manual(logits, labels)
    loss.backward()
    assert logits.grad is not None and not torch.any(torch.isnan(logits.grad))
_check("gradient flows back through logits", _t2)
''',
        },
    },
    {
        "id": "cross_entropy_backward",
        "title": "Cross-entropy backward (softmax grad)",
        "category": "Activations & Losses",
        "numpy": {
            "prompt": "Backprop through mean cross-entropy: return (loss, dlogits), the gradient of the MEAN loss w.r.t. the raw logits. Closed form: dlogits = (softmax(logits) - one_hot(labels)) / N.",
            "stub": '''def cross_entropy_backward(logits, labels):
    # TODO -- return (loss, dlogits)
    raise NotImplementedError''',
            "solution": '''def cross_entropy_backward(logits, labels):
    N, C = logits.shape
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(exp, axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(N), labels])
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits''',
            "tests": '''
def _t1():
    logits = np.array([[2.0,1.0,0.1],[0.1,0.2,3.0]])
    labels = np.array([0,2])
    loss, dlogits = cross_entropy_backward(logits, labels)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    ref_loss = -np.mean(np.log(probs[np.arange(2), labels]))
    assert np.isclose(loss, ref_loss, atol=1e-6)
    assert dlogits.shape == logits.shape
_check("loss matches independently-computed reference", _t1)

def _t2():
    rng = np.random.RandomState(0)
    logits = rng.randn(4,5)
    labels = rng.randint(0,5,size=4)
    _, dlogits = cross_entropy_backward(logits, labels)
    eps = 1e-5
    logits_plus = logits.copy(); logits_plus[1,2] += eps
    loss_plus, _ = cross_entropy_backward(logits_plus, labels)
    loss0, _ = cross_entropy_backward(logits, labels)
    numeric = (loss_plus - loss0) / eps
    assert np.isclose(numeric, dlogits[1,2], atol=1e-3)
_check("dlogits passes numerical gradient check", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors, computed by hand (no autograd, no F.cross_entropy) -- but it should exactly match autograd's own gradient through F.cross_entropy.",
            "stub": '''def cross_entropy_backward(logits, labels):
    # TODO -- return (loss, dlogits)
    raise NotImplementedError''',
            "solution": '''def cross_entropy_backward(logits, labels):
    N, C = logits.shape
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True)
    log_probs = shifted - torch.log(exp.sum(dim=1, keepdim=True))
    loss = -log_probs[torch.arange(N), labels].mean()
    dlogits = probs.clone()
    dlogits[torch.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    logits = torch.randn(4,5)
    labels = torch.randint(0,5,(4,))
    loss, dlogits = cross_entropy_backward(logits, labels)
    logits_ag = logits.clone().requires_grad_(True)
    ref_loss = torch.nn.functional.cross_entropy(logits_ag, labels)
    ref_loss.backward()
    assert torch.isclose(loss, ref_loss.detach(), atol=1e-5)
    assert torch.allclose(dlogits, logits_ag.grad, atol=1e-5)
_check("matches autograd through F.cross_entropy exactly", _t1)
''',
        },
    },
    {
        "id": "linear_layer",
        "title": "Linear layer forward/backward",
        "category": "Layers & Normalization",
        "numpy": {
            "prompt": "Manual forward+backward through y = X @ W + b. Return the tuple (out, dX, dW, db).",
            "stub": '''def linear_forward_backward(X, W, b, dout):
    # TODO -- return (out, dX, dW, db)
    raise NotImplementedError''',
            "solution": '''def linear_forward_backward(X, W, b, dout):
    out = X @ W + b
    dX = dout @ W.T
    dW = X.T @ dout
    db = dout.sum(axis=0)
    return out, dX, dW, db''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(1)
    X = rng.randn(4,3); W = rng.randn(3,2); b = rng.randn(2)
    dout = rng.randn(4,2)
    out, dX, dW, db = linear_forward_backward(X, W, b, dout)
    assert np.allclose(out, X@W+b, atol=1e-6)
    assert np.allclose(dX, dout@W.T, atol=1e-6)
    assert np.allclose(dW, X.T@dout, atol=1e-6)
    assert np.allclose(db, dout.sum(axis=0), atol=1e-6)
_check("out/dX/dW/db match closed form", _t1)

def _t2():
    rng = np.random.RandomState(1)
    X = rng.randn(4,3); W = rng.randn(3,2); b = rng.randn(2)
    dout = rng.randn(4,2)
    _, _, dW, _ = linear_forward_backward(X, W, b, dout)
    eps = 1e-5
    W_plus = W.copy(); W_plus[0,0] += eps
    numeric = np.sum(((X@W_plus+b)-(X@W+b))/eps * dout)
    assert np.isclose(numeric, dW[0,0], atol=1e-2)
_check("dW passes numerical gradient check", _t2)
''',
        },
        "torch": {
            "prompt": "Same algorithm with torch tensors — plain tensor ops, no autograd needed for this one. Return the tuple (out, dX, dW, db).",
            "stub": '''def linear_forward_backward(X, W, b, dout):
    # TODO -- return (out, dX, dW, db)
    raise NotImplementedError''',
            "solution": '''def linear_forward_backward(X, W, b, dout):
    out = X @ W + b
    dX = dout @ W.t()
    dW = X.t() @ dout
    db = dout.sum(dim=0)
    return out, dX, dW, db''',
            "tests": '''
def _t1():
    torch.manual_seed(1)
    X = torch.randn(4,3); W = torch.randn(3,2); b = torch.randn(2)
    dout = torch.randn(4,2)
    out, dX, dW, db = linear_forward_backward(X, W, b, dout)
    assert torch.allclose(out, X@W+b, atol=1e-6)
    assert torch.allclose(dX, dout@W.t(), atol=1e-6)
    assert torch.allclose(dW, X.t()@dout, atol=1e-6)
    assert torch.allclose(db, dout.sum(dim=0), atol=1e-6)
_check("out/dX/dW/db match closed form", _t1)
''',
        },
    },
    {
        "id": "batchnorm",
        "title": "Batch normalization",
        "category": "Layers & Normalization",
        "numpy": {
            "prompt": "BatchNorm forward, training mode (per-batch mean/var over axis 0). x:(N,D), gamma/beta:(D,).",
            "stub": '''def batchnorm_forward(x, gamma, beta, eps=1e-5):
    # TODO
    raise NotImplementedError''',
            "solution": '''def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(2)
    x = rng.randn(20,5)*5+3
    out = batchnorm_forward(x, np.ones(5), np.zeros(5))
    assert np.allclose(out.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(out.std(axis=0), 1, atol=1e-2)
_check("normalizes to mean 0 / std 1", _t1)

def _t2():
    rng = np.random.RandomState(2)
    x = rng.randn(20,5)*5+3
    out = batchnorm_forward(x, np.full(5,2.0), np.full(5,1.0))
    assert np.allclose(out.mean(axis=0), 1.0, atol=1e-5)
    assert np.allclose(out.std(axis=0), 2.0, atol=1e-2)
_check("gamma/beta shift & scale correctly", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors (unbiased=False variance, matching the training-time convention).",
            "stub": '''def batchnorm_forward(x, gamma, beta, eps=1e-5):
    # TODO
    raise NotImplementedError''',
            "solution": '''def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta''',
            "tests": '''
def _t1():
    torch.manual_seed(2)
    x = torch.randn(20,5)*5+3
    out = batchnorm_forward(x, torch.ones(5), torch.zeros(5))
    assert torch.allclose(out.mean(dim=0), torch.zeros(5), atol=1e-5)
    assert torch.allclose(out.std(dim=0, unbiased=False), torch.ones(5), atol=1e-2)
_check("normalizes to mean 0 / std 1", _t1)
''',
        },
    },
    {
        "id": "batchnorm_backward",
        "title": "BatchNorm backward",
        "category": "Layers & Normalization",
        "numpy": {
            "prompt": "Backprop through BatchNorm (training mode, per-batch stats). x,dout:(N,D), gamma/beta:(D,). Return the tuple (out, dx, dgamma, dbeta).",
            "stub": '''def batchnorm_backward(x, gamma, beta, dout, eps=1e-5):
    # TODO -- return (out, dx, dgamma, dbeta)
    raise NotImplementedError''',
            "solution": '''def batchnorm_backward(x, gamma, beta, dout, eps=1e-5):
    N, D = x.shape
    mu = x.mean(axis=0)
    xc = x - mu
    var = (xc**2).mean(axis=0)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = xc * std_inv
    out = gamma * x_hat + beta

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * xc * -0.5 * std_inv**3, axis=0)
    dmu = np.sum(dx_hat * -std_inv, axis=0) + dvar * np.mean(-2 * xc, axis=0)
    dx = dx_hat * std_inv + dvar * 2 * xc / N + dmu / N
    return out, dx, dgamma, dbeta''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(7)
    N, D = 6, 4
    x = rng.randn(N, D) * 3 + 1
    gamma = rng.randn(D); beta = rng.randn(D); dout = rng.randn(N, D)
    out, dx, dgamma, dbeta = batchnorm_backward(x, gamma, beta, dout)
    assert dx.shape == x.shape
    assert dgamma.shape == (D,) and dbeta.shape == (D,)
_check("shapes correct", _t1)

def _t2():
    rng = np.random.RandomState(7)
    N, D = 6, 4
    x = rng.randn(N, D) * 3 + 1
    gamma = rng.randn(D); beta = rng.randn(D); dout = rng.randn(N, D)
    out, dx, dgamma, dbeta = batchnorm_backward(x, gamma, beta, dout)
    def bn_fwd(xa, gammaa, betaa, eps=1e-5):
        mu = xa.mean(axis=0)
        var = ((xa - mu) ** 2).mean(axis=0)
        return gammaa * (xa - mu) / np.sqrt(var + eps) + betaa
    eps = 1e-5
    x_plus = x.copy(); x_plus[2,1] += eps
    numeric_dx = np.sum((bn_fwd(x_plus, gamma, beta) - bn_fwd(x, gamma, beta)) / eps * dout)
    assert np.isclose(numeric_dx, dx[2,1], atol=1e-2)
    gamma_plus = gamma.copy(); gamma_plus[1] += eps
    numeric_dgamma = np.sum((bn_fwd(x, gamma_plus, beta) - bn_fwd(x, gamma, beta)) / eps * dout)
    assert np.isclose(numeric_dgamma, dgamma[1], atol=1e-2)
_check("dx & dgamma pass numerical gradient check", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors, computed by hand -- but it should exactly match autograd's own gradient through the same formula. Return the tuple (out, dx, dgamma, dbeta).",
            "stub": '''def batchnorm_backward(x, gamma, beta, dout, eps=1e-5):
    # TODO -- return (out, dx, dgamma, dbeta)
    raise NotImplementedError''',
            "solution": '''def batchnorm_backward(x, gamma, beta, dout, eps=1e-5):
    N, D = x.shape
    mu = x.mean(dim=0)
    xc = x - mu
    var = (xc**2).mean(dim=0)
    std_inv = 1.0 / torch.sqrt(var + eps)
    x_hat = xc * std_inv
    out = gamma * x_hat + beta

    dgamma = torch.sum(dout * x_hat, dim=0)
    dbeta = torch.sum(dout, dim=0)
    dx_hat = dout * gamma
    dvar = torch.sum(dx_hat * xc * -0.5 * std_inv**3, dim=0)
    dmu = torch.sum(dx_hat * -std_inv, dim=0) + dvar * torch.mean(-2 * xc, dim=0)
    dx = dx_hat * std_inv + dvar * 2 * xc / N + dmu / N
    return out, dx, dgamma, dbeta''',
            "tests": '''
def _t1():
    torch.manual_seed(7)
    N, D = 6, 4
    x = torch.randn(N, D, requires_grad=True)
    gamma = torch.randn(D, requires_grad=True)
    beta = torch.randn(D, requires_grad=True)
    dout = torch.randn(N, D)
    out, dx, dgamma, dbeta = batchnorm_backward(x.detach(), gamma.detach(), beta.detach(), dout)

    mu = x.mean(dim=0)
    var = ((x - mu) ** 2).mean(dim=0)
    x_hat = (x - mu) / torch.sqrt(var + 1e-5)
    out_ref = gamma * x_hat + beta
    out_ref.backward(dout)

    assert torch.allclose(out, out_ref.detach(), atol=1e-5)
    assert torch.allclose(dx, x.grad, atol=1e-4)
    assert torch.allclose(dgamma, gamma.grad, atol=1e-4)
    assert torch.allclose(dbeta, beta.grad, atol=1e-4)
_check("matches autograd through the same formula exactly", _t1)
''',
        },
    },
    {
        "id": "layernorm",
        "title": "Layer normalization",
        "category": "Layers & Normalization",
        "numpy": {
            "prompt": "LayerNorm over the last axis (per-sample, not per-batch). x:(...,D), gamma/beta:(D,).",
            "stub": '''def layer_norm_manual(x, gamma, beta, eps=1e-5):
    # TODO
    raise NotImplementedError''',
            "solution": '''def layer_norm_manual(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(4)
    x = rng.randn(6,10)*10+3
    out = layer_norm_manual(x, np.ones(10), np.zeros(10))
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(out.std(axis=-1), 1, atol=1e-2)
_check("normalizes each row to mean 0 / std 1", _t1)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. No nn.LayerNorm / F.layer_norm.",
            "stub": '''def layer_norm_manual(x, gamma, beta, eps=1e-5):
    # TODO
    raise NotImplementedError''',
            "solution": '''def layer_norm_manual(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta''',
            "tests": '''
def _t1():
    x = torch.randn(6, 10) * 10 + 3
    out = layer_norm_manual(x, torch.ones(10), torch.zeros(10))
    assert torch.allclose(out.mean(dim=-1), torch.zeros(6), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(6), atol=1e-2)
_check("normalizes each row to mean 0 / std 1", _t1)
''',
        },
    },
    {
        "id": "kmeans",
        "title": "K-means clustering",
        "category": "Classical ML & Training",
        "numpy": {
            "prompt": "Lloyd's algorithm. Init centroids by randomly picking k points (RandomState(seed)). Return (centroids, labels).",
            "stub": '''def kmeans(X, k, num_iters=100, seed=0):
    # TODO
    raise NotImplementedError''',
            "solution": '''def kmeans(X, k, num_iters=100, seed=0):
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[init_idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for it in range(num_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
    return centroids, labels''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(3)
    c1 = rng.randn(25,2)+np.array([10,10])
    c2 = rng.randn(25,2)+np.array([-10,-10])
    X = np.vstack([c1,c2])
    centroids, labels = kmeans(X, k=2, num_iters=50, seed=0)
    assert centroids.shape == (2,2)
    assert np.linalg.norm(centroids[0]-centroids[1]) > 10
_check("finds two well-separated clusters", _t1)
''',
        },
        "torch": {
            "prompt": "Same algorithm with torch tensors (torch.cdist is fine to use for distances).",
            "stub": '''def kmeans(X, k, num_iters=100, seed=0):
    # TODO
    raise NotImplementedError''',
            "solution": '''def kmeans(X, k, num_iters=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.size(0), generator=g)[:k]
    centroids = X[idx].clone()
    labels = torch.zeros(X.size(0), dtype=torch.long)
    for it in range(num_iters):
        dists = torch.cdist(X, centroids)
        new_labels = dists.argmin(dim=1)
        if it > 0 and torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(dim=0)
    return centroids, labels''',
            "tests": '''
def _t1():
    torch.manual_seed(3)
    c1 = torch.randn(25,2)+torch.tensor([10.,10.])
    c2 = torch.randn(25,2)+torch.tensor([-10.,-10.])
    X = torch.cat([c1,c2])
    centroids, labels = kmeans(X, k=2, num_iters=50, seed=0)
    assert centroids.shape == (2,2)
    assert torch.norm(centroids[0]-centroids[1]) > 10
_check("finds two well-separated clusters", _t1)
''',
        },
    },
    {
        "id": "conv2d",
        "title": "Naive 2D convolution",
        "category": "CNNs",
        "numpy": {
            "prompt": "Naive 2D cross-correlation, single channel. X:(H,W_in), kernel W:(kh,kw).",
            "stub": '''def conv2d_naive(X, W, stride=1, padding=0):
    # TODO
    raise NotImplementedError''',
            "solution": '''def conv2d_naive(X, W, stride=1, padding=0):
    if padding > 0:
        X = np.pad(X, ((padding, padding), (padding, padding)), mode="constant")
    H, Wd = X.shape
    kh, kw = W.shape
    out_h = (H - kh) // stride + 1
    out_w = (Wd - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r0, c0 = i*stride, j*stride
            out[i,j] = np.sum(X[r0:r0+kh, c0:c0+kw] * W)
    return out''',
            "tests": '''
def _t1():
    X = np.arange(16).reshape(4,4).astype(float)
    W = np.array([[1.0,0.0],[0.0,1.0]])
    out = conv2d_naive(X, W, stride=1, padding=0)
    assert out.shape == (3,3)
    assert np.isclose(out[0,0], X[0,0]+X[1,1])
_check("correct output & values, stride=1", _t1)

def _t2():
    X = np.arange(16).reshape(4,4).astype(float)
    W = np.array([[1.0,0.0],[0.0,1.0]])
    assert conv2d_naive(X, W, stride=2, padding=0).shape == (2,2)
    assert conv2d_naive(X, W, stride=1, padding=1).shape == (5,5)
_check("stride and padding change output shape correctly", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. No F.conv2d — pad and slide the window manually.",
            "stub": '''def conv2d_naive(X, W, stride=1, padding=0):
    # TODO
    raise NotImplementedError''',
            "solution": '''def conv2d_naive(X, W, stride=1, padding=0):
    if padding > 0:
        H, Wd = X.shape
        padded = torch.zeros(H+2*padding, Wd+2*padding)
        padded[padding:padding+H, padding:padding+Wd] = X
        X = padded
    H, Wd = X.shape
    kh, kw = W.shape
    out_h = (H - kh) // stride + 1
    out_w = (Wd - kw) // stride + 1
    out = torch.zeros(out_h, out_w)
    for i in range(out_h):
        for j in range(out_w):
            r0, c0 = i*stride, j*stride
            out[i,j] = (X[r0:r0+kh, c0:c0+kw] * W).sum()
    return out''',
            "tests": '''
def _t1():
    X = torch.arange(16).reshape(4,4).float()
    W = torch.tensor([[1.0,0.0],[0.0,1.0]])
    out = conv2d_naive(X, W, stride=1, padding=0)
    assert out.shape == (3,3)
    assert torch.isclose(out[0,0], X[0,0]+X[1,1])
    assert conv2d_naive(X, W, stride=2, padding=0).shape == (2,2)
    assert conv2d_naive(X, W, stride=1, padding=1).shape == (5,5)
_check("correct output, shapes with stride/padding", _t1)
''',
        },
    },
    {
        "id": "conv2d_channels",
        "title": "2D convolution with channels",
        "category": "CNNs",
        "numpy": {
            "prompt": "Every conv exercise so far has been single-channel -- real CNNs never are. Naive 2D cross-correlation with multiple input/output channels (this is literally what nn.Conv2d computes): X:(C_in,H,W), kernel:(C_out,C_in,kh,kw), bias:(C_out,). For each output channel, sum the single-channel correlation across every input channel, then add that channel's own bias.",
            "stub": '''def conv2d_channels(X, kernel, bias, stride=1, padding=0):
    # TODO
    raise NotImplementedError''',
            "solution": '''def conv2d_channels(X, kernel, bias, stride=1, padding=0):
    C_in, H, W = X.shape
    C_out = kernel.shape[0]
    if padding > 0:
        X = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
        H, W = X.shape[1], X.shape[2]
    kh, kw = kernel.shape[2], kernel.shape[3]
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((C_out, out_h, out_w))
    for co in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                r0, c0 = i * stride, j * stride
                patch = X[:, r0:r0+kh, c0:c0+kw]
                out[co, i, j] = np.sum(patch * kernel[co]) + bias[co]
    return out''',
            "tests": '''
def _t1():
    X = np.zeros((2, 3, 3))
    X[0] = np.arange(9).reshape(3, 3)
    X[1] = np.full((3, 3), 10.0)
    kernel = np.zeros((1, 2, 2, 2))
    kernel[0, 0] = [[1, 0], [0, 1]]
    kernel[0, 1] = [[1, 1], [1, 1]]
    bias = np.array([5.0])
    out = conv2d_channels(X, kernel, bias)
    expected = np.array([[[49, 51], [55, 57]]])
    assert np.allclose(out, expected)
_check("matches hand-computed multi-channel correlation", _t1)

def _t2():
    rng = np.random.RandomState(0)
    X = rng.randn(3, 6, 6)
    kernel = rng.randn(4, 3, 3, 3)
    bias = rng.randn(4)
    out = conv2d_channels(X, kernel, bias, stride=2, padding=1)
    assert out.shape == (4, 3, 3)
    Xp = np.zeros((3, 8, 8))
    Xp[:, 1:7, 1:7] = X
    ref = np.zeros((4, 3, 3))
    for co in range(4):
        for ci in range(3):
            for i in range(3):
                for j in range(3):
                    r0, c0 = i*2, j*2
                    ref[co, i, j] += np.sum(Xp[ci, r0:r0+3, c0:c0+3] * kernel[co, ci])
        ref[co] += bias[co]
    assert np.allclose(out, ref, atol=1e-6)
_check("matches an independently-computed channel-by-channel reference with stride & padding", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. No F.conv2d / nn.Conv2d -- but it should exactly match F.conv2d's output.",
            "stub": '''def conv2d_channels(X, kernel, bias, stride=1, padding=0):
    # TODO
    raise NotImplementedError''',
            "solution": '''def conv2d_channels(X, kernel, bias, stride=1, padding=0):
    C_in, H, W = X.shape
    C_out = kernel.shape[0]
    if padding > 0:
        Xp = torch.zeros(C_in, H + 2*padding, W + 2*padding, dtype=X.dtype)
        Xp[:, padding:padding+H, padding:padding+W] = X
        X = Xp
        H, W = X.shape[1], X.shape[2]
    kh, kw = kernel.shape[2], kernel.shape[3]
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = torch.zeros(C_out, out_h, out_w, dtype=X.dtype)
    for co in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                r0, c0 = i * stride, j * stride
                patch = X[:, r0:r0+kh, c0:c0+kw]
                out[co, i, j] = (patch * kernel[co]).sum() + bias[co]
    return out''',
            "tests": '''
def _t1():
    X = torch.zeros(2, 3, 3)
    X[0] = torch.arange(9).reshape(3, 3).float()
    X[1] = torch.full((3, 3), 10.0)
    kernel = torch.zeros(1, 2, 2, 2)
    kernel[0, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kernel[0, 1] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    bias = torch.tensor([5.0])
    out = conv2d_channels(X, kernel, bias)
    expected = torch.tensor([[[49.0, 51.0], [55.0, 57.0]]])
    assert torch.allclose(out, expected)
_check("matches hand-computed multi-channel correlation", _t1)

def _t2():
    torch.manual_seed(0)
    X = torch.randn(3, 6, 6)
    kernel = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    out = conv2d_channels(X, kernel, bias, stride=2, padding=1)
    ref = torch.nn.functional.conv2d(X.unsqueeze(0), kernel, bias=bias, stride=2, padding=1).squeeze(0)
    assert torch.allclose(out, ref, atol=1e-4)
_check("matches real F.conv2d with stride & padding", _t2)
''',
        },
    },
    {
        "id": "conv2d_backward",
        "title": "2D convolution backward",
        "category": "CNNs",
        "numpy": {
            "prompt": "Backprop through a valid (no padding), stride-1, single-channel 2D correlation. X:(H,W), kernel:(kh,kw), dout:(H-kh+1,W-kw+1). Return the tuple (out, dX, dW).",
            "stub": '''def conv2d_backward(X, W, dout):
    # TODO -- return (out, dX, dW)
    raise NotImplementedError''',
            "solution": '''def conv2d_backward(X, W, dout):
    kh, kw = W.shape
    out_h, out_w = dout.shape
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(X[i:i+kh, j:j+kw] * W)
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    for i in range(out_h):
        for j in range(out_w):
            dX[i:i+kh, j:j+kw] += dout[i, j] * W
            dW += dout[i, j] * X[i:i+kh, j:j+kw]
    return out, dX, dW''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(5)
    X = rng.randn(5,5); W = rng.randn(3,3); dout = rng.randn(3,3)
    out, dX, dW = conv2d_backward(X, W, dout)
    ref_out = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            ref_out[i,j] = np.sum(X[i:i+3,j:j+3]*W)
    assert np.allclose(out, ref_out, atol=1e-6)
    assert dX.shape == X.shape and dW.shape == W.shape
_check("forward output & gradient shapes correct", _t1)

def _t2():
    rng = np.random.RandomState(5)
    X = rng.randn(5,5); W = rng.randn(3,3); dout = rng.randn(3,3)
    out, dX, dW = conv2d_backward(X, W, dout)
    def fwd(Xa, Wa):
        oh, ow = Xa.shape[0]-Wa.shape[0]+1, Xa.shape[1]-Wa.shape[1]+1
        o = np.zeros((oh,ow))
        for i in range(oh):
            for j in range(ow):
                o[i,j] = np.sum(Xa[i:i+Wa.shape[0], j:j+Wa.shape[1]]*Wa)
        return o
    eps = 1e-5
    W_plus = W.copy(); W_plus[1,1] += eps
    numeric_dW = np.sum((fwd(X,W_plus)-fwd(X,W))/eps * dout)
    assert np.isclose(numeric_dW, dW[1,1], atol=1e-2)
    X_plus = X.copy(); X_plus[2,2] += eps
    numeric_dX = np.sum((fwd(X_plus,W)-fwd(X,W))/eps * dout)
    assert np.isclose(numeric_dX, dX[2,2], atol=1e-2)
_check("dW/dX pass numerical gradient check", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors, computed by hand (no autograd inside your function) -- but it should exactly match real autograd gradients through the same convolution. Return the tuple (out, dX, dW).",
            "stub": '''def conv2d_backward(X, W, dout):
    # TODO -- return (out, dX, dW)
    raise NotImplementedError''',
            "solution": '''def conv2d_backward(X, W, dout):
    kh, kw = W.shape
    out_h, out_w = dout.shape
    out = torch.zeros(out_h, out_w)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = (X[i:i+kh, j:j+kw] * W).sum()
    dX = torch.zeros_like(X)
    dW = torch.zeros_like(W)
    for i in range(out_h):
        for j in range(out_w):
            dX[i:i+kh, j:j+kw] += dout[i, j] * W
            dW += dout[i, j] * X[i:i+kh, j:j+kw]
    return out, dX, dW''',
            "tests": '''
def _t1():
    torch.manual_seed(5)
    X = torch.randn(5,5, requires_grad=True); W = torch.randn(3,3, requires_grad=True)
    dout = torch.randn(3,3)
    out, dX, dW = conv2d_backward(X.detach(), W.detach(), dout)
    out_ref = torch.zeros(3,3)
    for i in range(3):
        for j in range(3):
            out_ref[i,j] = (X[i:i+3,j:j+3]*W).sum()
    out_ref.backward(dout)
    assert torch.allclose(out, out_ref.detach(), atol=1e-5)
    assert torch.allclose(dW, W.grad, atol=1e-5)
    assert torch.allclose(dX, X.grad, atol=1e-5)
_check("matches autograd exactly", _t1)
''',
        },
    },
    {
        "id": "cnn_block",
        "title": "CNN block (conv + ReLU + maxpool)",
        "category": "CNNs",
        "numpy": {
            "prompt": "A tiny CNN block: naive valid 2D conv (single channel, stride 1) + bias, then ReLU, then non-overlapping 2x2 max-pool (stride 2, drop any leftover row/col). X:(H,W), kernel:(kh,kw), bias: scalar.",
            "stub": '''def cnn_block_forward(X, kernel, bias):
    # TODO
    raise NotImplementedError''',
            "solution": '''def cnn_block_forward(X, kernel, bias):
    H, W = X.shape
    kh, kw = kernel.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    conv = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            conv[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel) + bias
    relu = np.maximum(conv, 0)
    ph, pw = out_h // 2, out_w // 2
    pooled = np.zeros((ph, pw))
    for i in range(ph):
        for j in range(pw):
            pooled[i, j] = np.max(relu[2*i:2*i+2, 2*j:2*j+2])
    return pooled''',
            "tests": '''
def _t1():
    X = np.arange(25).reshape(5,5).astype(float)
    kernel = np.array([[1.0,0.0],[0.0,1.0]])
    out = cnn_block_forward(X, kernel, -20.0)
    assert out.shape == (2,2)
    assert np.allclose(out, [[0,2],[18,22]])
_check("matches hand-computed conv+relu+pool", _t1)

def _t2():
    rng = np.random.RandomState(0)
    X = rng.randn(6,6); kernel = rng.randn(3,3)
    out = cnn_block_forward(X, kernel, 0.5)
    assert out.shape == (2,2)
    assert np.all(out >= 0)
_check("output shape correct & non-negative after ReLU", _t2)
''',
        },
        "torch": {
            "prompt": "Same block with torch tensors. No nn.Conv2d / nn.MaxPool2d / F.conv2d / F.max_pool2d.",
            "stub": '''def cnn_block_forward(X, kernel, bias):
    # TODO
    raise NotImplementedError''',
            "solution": '''def cnn_block_forward(X, kernel, bias):
    H, W = X.shape
    kh, kw = kernel.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    conv = torch.zeros(out_h, out_w)
    for i in range(out_h):
        for j in range(out_w):
            conv[i, j] = (X[i:i+kh, j:j+kw] * kernel).sum() + bias
    relu = torch.clamp(conv, min=0)
    ph, pw = out_h // 2, out_w // 2
    pooled = torch.zeros(ph, pw)
    for i in range(ph):
        for j in range(pw):
            pooled[i, j] = relu[2*i:2*i+2, 2*j:2*j+2].max()
    return pooled''',
            "tests": '''
def _t1():
    X = torch.arange(25).reshape(5,5).float()
    kernel = torch.tensor([[1.0,0.0],[0.0,1.0]])
    out = cnn_block_forward(X, kernel, -20.0)
    assert out.shape == (2,2)
    assert torch.allclose(out, torch.tensor([[0.,2.],[18.,22.]]))
_check("matches hand-computed conv+relu+pool", _t1)

def _t2():
    torch.manual_seed(0)
    X = torch.randn(6,6); kernel = torch.randn(3,3)
    out = cnn_block_forward(X, kernel, 0.5)
    assert out.shape == (2,2)
    assert torch.all(out >= 0)
_check("output shape correct & non-negative after ReLU", _t2)
''',
        },
    },
    {
        "id": "resnet_block",
        "title": "ResNet residual block",
        "category": "CNNs",
        "numpy": None,
        "torch": {
            "prompt": "A basic ResNet block: conv-bn-relu, conv-bn, add the shortcut, then a final relu. conv1/bn1/conv2/bn2 and (if the shape changes) a projection self.shortcut are already built for you in __init__ -- self.shortcut is None when in_channels==out_channels and stride==1 (identity shortcut), otherwise it's a 1x1 conv+BN to match shape. No nn.Sequential shortcuts inside forward -- that's the part you're wiring.",
            "stub": '''class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        # TODO -- conv1/bn1/relu -> conv2/bn2 -> add the (projected) shortcut -> relu
        raise NotImplementedError''',
            "solution": '''class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)''',
            "tests": '''
def _t1():
    block = ResNetBlock(in_channels=4, out_channels=4, stride=1)
    assert block.shortcut is None
    x = torch.randn(2, 4, 8, 8)
    out = block(x)
    assert out.shape == (2, 4, 8, 8)
    assert torch.all(out >= 0)
_check("same-shape block needs no projection shortcut", _t1)

def _t2():
    block = ResNetBlock(in_channels=4, out_channels=8, stride=2)
    assert block.shortcut is not None
    x = torch.randn(2, 4, 16, 16)
    out = block(x)
    assert out.shape == (2, 8, 8, 8)
_check("downsampling block projects the shortcut & halves spatial dims", _t2)

def _t3():
    torch.manual_seed(0)
    block = ResNetBlock(in_channels=4, out_channels=4, stride=1)
    block.eval()  # deterministic BatchNorm (running stats) for a stable comparison
    x = torch.randn(2, 4, 8, 8)
    out = block(x)
    identity = x if block.shortcut is None else block.shortcut(x)
    ref = torch.relu(block.bn1(block.conv1(x)))
    ref = block.bn2(block.conv2(ref))
    ref = torch.relu(ref + identity)
    assert torch.allclose(out, ref, atol=1e-5)
_check("matches conv-bn-relu, conv-bn, + shortcut, final relu", _t3)
''',
        },
    },
    {
        "id": "rnn",
        "title": "Vanilla RNN cell",
        "category": "Sequence Models",
        "numpy": {
            "prompt": "One step of a vanilla RNN cell: h' = tanh(x @ weight_ih.T + bias_ih + h_prev @ weight_hh.T + bias_hh). Shapes match PyTorch's nn.RNNCell: weight_ih (H,D), weight_hh (H,H), bias_ih/bias_hh (H,).",
            "stub": '''def rnn_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    # TODO
    raise NotImplementedError''',
            "solution": '''def rnn_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    return np.tanh(x @ weight_ih.T + bias_ih + h_prev @ weight_hh.T + bias_hh)''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    N, D, H = 4, 3, 5
    x = rng.randn(N, D); h = rng.randn(N, H)
    weight_ih = rng.randn(H, D); weight_hh = rng.randn(H, H)
    bias_ih = rng.randn(H); bias_hh = rng.randn(H)
    out = rnn_cell_forward(x, h, weight_ih, weight_hh, bias_ih, bias_hh)
    ref = np.tanh(x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh)
    assert out.shape == (N, H)
    assert np.allclose(out, ref, atol=1e-6)
_check("matches closed-form tanh recurrence", _t1)

def _t2():
    N, D, H = 2, 3, 4
    z = lambda *s: np.zeros(s)
    out = rnn_cell_forward(z(N, D), z(N, H), z(H, D), z(H, H), z(H), z(H))
    assert np.allclose(out, 0.0, atol=1e-8)
_check("all-zero inputs/weights give tanh(0) = 0", _t2)
''',
        },
        "torch": {
            "prompt": "Same cell, matching nn.RNNCell's exact parameter layout so it can be checked against the real thing. No nn.RNNCell in your implementation.",
            "stub": '''def rnn_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    # TODO
    raise NotImplementedError''',
            "solution": '''def rnn_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    return torch.tanh(x @ weight_ih.t() + bias_ih + h_prev @ weight_hh.t() + bias_hh)''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    N, D, H = 4, 3, 5
    cell = nn.RNNCell(D, H)
    x = torch.randn(N, D); h = torch.randn(N, H)
    out = rnn_cell_forward(x, h, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    ref = cell(x, h)
    assert torch.allclose(out, ref, atol=1e-5)
_check("matches nn.RNNCell exactly", _t1)

def _t2():
    torch.manual_seed(0)
    N, D, H = 3, 2, 4
    weight_ih = torch.randn(H, D, requires_grad=True); weight_hh = torch.randn(H, H, requires_grad=True)
    bias_ih = torch.randn(H, requires_grad=True); bias_hh = torch.randn(H, requires_grad=True)
    x = torch.randn(N, D); h0 = torch.zeros(N, H)
    out = rnn_cell_forward(x, h0, weight_ih, weight_hh, bias_ih, bias_hh)
    out.sum().backward()
    assert weight_ih.grad is not None and not torch.any(torch.isnan(weight_ih.grad))
_check("gradient flows back to weight_ih", _t2)
''',
        },
    },
    {
        "id": "lstm",
        "title": "LSTM cell",
        "category": "Sequence Models",
        "numpy": {
            "prompt": "One step of an LSTM cell, matching PyTorch's nn.LSTMCell parameter layout: weight_ih (4H,D), weight_hh (4H,H), bias_ih/bias_hh (4H,), gates stacked in order [i,f,g,o]. i,f,o use sigmoid; g uses tanh. c' = f*c_prev + i*g; h' = o*tanh(c').",
            "stub": '''def lstm_cell_forward(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    # TODO -- return (h_next, c_next)
    raise NotImplementedError''',
            "solution": '''def lstm_cell_forward(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    H = h_prev.shape[1]
    gates = x @ weight_ih.T + bias_ih + h_prev @ weight_hh.T + bias_hh
    i = 1 / (1 + np.exp(-gates[:, 0:H]))
    f = 1 / (1 + np.exp(-gates[:, H:2*H]))
    g = np.tanh(gates[:, 2*H:3*H])
    o = 1 / (1 + np.exp(-gates[:, 3*H:4*H]))
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    N, D, H = 4, 3, 5
    x = rng.randn(N, D); h0 = rng.randn(N, H); c0 = rng.randn(N, H)
    weight_ih = rng.randn(4*H, D); weight_hh = rng.randn(4*H, H)
    bias_ih = rng.randn(4*H); bias_hh = rng.randn(4*H)
    h, c = lstm_cell_forward(x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh)
    gates = x @ weight_ih.T + bias_ih + h0 @ weight_hh.T + bias_hh
    gi = 1/(1+np.exp(-gates[:,0:H])); gf = 1/(1+np.exp(-gates[:,H:2*H]))
    gg = np.tanh(gates[:,2*H:3*H]); go = 1/(1+np.exp(-gates[:,3*H:4*H]))
    c_ref = gf*c0 + gi*gg; h_ref = go*np.tanh(c_ref)
    assert h.shape == (N, H) and c.shape == (N, H)
    assert np.allclose(h, h_ref, atol=1e-6) and np.allclose(c, c_ref, atol=1e-6)
_check("matches closed-form gate computation", _t1)

def _t2():
    N, D, H = 2, 3, 4
    z = lambda *s: np.zeros(s)
    h, c = lstm_cell_forward(z(N, D), z(N, H), z(N, H), z(4*H, D), z(4*H, H), z(4*H), z(4*H))
    assert np.allclose(c, 0.0, atol=1e-8) and np.allclose(h, 0.0, atol=1e-8)
_check("all-zero weights/state keep the cell state at zero", _t2)
''',
        },
        "torch": {
            "prompt": "Same, matching nn.LSTMCell's exact parameter layout so it can be checked against the real thing. No nn.LSTMCell in your implementation.",
            "stub": '''def lstm_cell_forward(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    # TODO -- return (h_next, c_next)
    raise NotImplementedError''',
            "solution": '''def lstm_cell_forward(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    H = h_prev.size(1)
    gates = x @ weight_ih.t() + bias_ih + h_prev @ weight_hh.t() + bias_hh
    i = torch.sigmoid(gates[:, 0:H])
    f = torch.sigmoid(gates[:, H:2*H])
    g = torch.tanh(gates[:, 2*H:3*H])
    o = torch.sigmoid(gates[:, 3*H:4*H])
    c = f * c_prev + i * g
    h = o * torch.tanh(c)
    return h, c''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    N, D, H = 4, 3, 5
    cell = nn.LSTMCell(D, H)
    x = torch.randn(N, D); h0 = torch.randn(N, H); c0 = torch.randn(N, H)
    h, c = lstm_cell_forward(x, h0, c0, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    h_ref, c_ref = cell(x, (h0, c0))
    assert torch.allclose(h, h_ref, atol=1e-5) and torch.allclose(c, c_ref, atol=1e-5)
_check("matches nn.LSTMCell exactly", _t1)

def _t2():
    torch.manual_seed(0)
    N, D, H = 3, 2, 4
    weight_ih = torch.randn(4*H, D, requires_grad=True); weight_hh = torch.randn(4*H, H, requires_grad=True)
    bias_ih = torch.randn(4*H, requires_grad=True); bias_hh = torch.randn(4*H, requires_grad=True)
    x = torch.randn(N, D); h0 = torch.zeros(N, H); c0 = torch.zeros(N, H)
    h, c = lstm_cell_forward(x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh)
    (h.sum() + c.sum()).backward()
    assert weight_ih.grad is not None and not torch.any(torch.isnan(weight_ih.grad))
_check("gradient flows back to weight_ih through the gates and cell state", _t2)
''',
        },
    },
    {
        "id": "gru",
        "title": "GRU cell",
        "category": "Sequence Models",
        "numpy": {
            "prompt": "One step of a GRU cell matching nn.GRUCell's layout: weight_ih (3H,D), weight_hh (3H,H), bias_ih/bias_hh (3H,), gates stacked [r,z,n]. r,z use sigmoid. n = tanh(i_n + r*h_n) -- keep the input-side and hidden-side matmuls separate (don't sum them before splitting into gates), since r only gates the hidden contribution to n.",
            "stub": '''def gru_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    # TODO
    raise NotImplementedError''',
            "solution": '''def gru_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    H = h_prev.shape[1]
    gi = x @ weight_ih.T + bias_ih
    gh = h_prev @ weight_hh.T + bias_hh
    i_r, i_z, i_n = gi[:, 0:H], gi[:, H:2*H], gi[:, 2*H:3*H]
    h_r, h_z, h_n = gh[:, 0:H], gh[:, H:2*H], gh[:, 2*H:3*H]
    r = 1 / (1 + np.exp(-(i_r + h_r)))
    z = 1 / (1 + np.exp(-(i_z + h_z)))
    n = np.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h_prev''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    N, D, H = 4, 3, 5
    x = rng.randn(N, D); h0 = rng.randn(N, H)
    weight_ih = rng.randn(3*H, D); weight_hh = rng.randn(3*H, H)
    bias_ih = rng.randn(3*H); bias_hh = rng.randn(3*H)
    out = gru_cell_forward(x, h0, weight_ih, weight_hh, bias_ih, bias_hh)
    gi = x @ weight_ih.T + bias_ih; gh = h0 @ weight_hh.T + bias_hh
    i_r, i_z, i_n = gi[:,0:H], gi[:,H:2*H], gi[:,2*H:3*H]
    h_r, h_z, h_n = gh[:,0:H], gh[:,H:2*H], gh[:,2*H:3*H]
    r = 1/(1+np.exp(-(i_r+h_r))); z = 1/(1+np.exp(-(i_z+h_z))); n = np.tanh(i_n + r*h_n)
    ref = (1-z)*n + z*h0
    assert out.shape == (N, H)
    assert np.allclose(out, ref, atol=1e-6)
_check("matches closed-form gate computation", _t1)

def _t2():
    N, D, H = 2, 3, 4
    weight_ih = np.zeros((3*H, D)); weight_hh = np.zeros((3*H, H))
    bias_ih = np.zeros(3*H); bias_hh = np.zeros(3*H)
    bias_ih[H:2*H] = 50.0  # push the update gate z -> 1
    h_prev = np.random.RandomState(1).randn(N, H)
    out = gru_cell_forward(np.zeros((N, D)), h_prev, weight_ih, weight_hh, bias_ih, bias_hh)
    assert np.allclose(out, h_prev, atol=1e-6)
_check("update gate saturated to 1 leaves hidden state unchanged", _t2)
''',
        },
        "torch": {
            "prompt": "Same, matching nn.GRUCell's exact parameter layout so it can be checked against the real thing. No nn.GRUCell in your implementation.",
            "stub": '''def gru_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    # TODO
    raise NotImplementedError''',
            "solution": '''def gru_cell_forward(x, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    H = h_prev.size(1)
    gi = x @ weight_ih.t() + bias_ih
    gh = h_prev @ weight_hh.t() + bias_hh
    i_r, i_z, i_n = gi[:, 0:H], gi[:, H:2*H], gi[:, 2*H:3*H]
    h_r, h_z, h_n = gh[:, 0:H], gh[:, H:2*H], gh[:, 2*H:3*H]
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h_prev''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    N, D, H = 4, 3, 5
    cell = nn.GRUCell(D, H)
    x = torch.randn(N, D); h0 = torch.randn(N, H)
    out = gru_cell_forward(x, h0, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    ref = cell(x, h0)
    assert torch.allclose(out, ref, atol=1e-5)
_check("matches nn.GRUCell exactly", _t1)

def _t2():
    torch.manual_seed(0)
    N, D, H = 3, 2, 4
    weight_ih = torch.randn(3*H, D, requires_grad=True); weight_hh = torch.randn(3*H, H, requires_grad=True)
    bias_ih = torch.randn(3*H, requires_grad=True); bias_hh = torch.randn(3*H, requires_grad=True)
    x = torch.randn(N, D); h0 = torch.zeros(N, H)
    out = gru_cell_forward(x, h0, weight_ih, weight_hh, bias_ih, bias_hh)
    out.sum().backward()
    assert weight_ih.grad is not None and not torch.any(torch.isnan(weight_ih.grad))
_check("gradient flows back to weight_ih", _t2)
''',
        },
    },
    {
        "id": "prf1",
        "title": "Precision / recall / F1",
        "category": "Classical ML & Training",
        "numpy": {
            "prompt": "Binary precision, recall, F1 from 0/1 arrays. 0.0 on zero-division.",
            "stub": '''def precision_recall_f1(y_true, y_pred):
    # TODO
    raise NotImplementedError''',
            "solution": '''def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0
    return float(precision), float(recall), float(f1)''',
            "tests": '''
def _t1():
    y_true = np.array([1,1,0,0,1]); y_pred = np.array([1,0,0,0,1])
    p,r,f1 = precision_recall_f1(y_true, y_pred)
    assert np.isclose(p, 1.0) and np.isclose(r, 2/3, atol=1e-6)
    assert np.isclose(f1, 2*1.0*(2/3)/(1.0+2/3), atol=1e-6)
_check("matches hand-computed precision/recall/f1", _t1)

def _t2():
    p,r,f1 = precision_recall_f1(np.array([1,0]), np.array([0,0]))
    assert p==0.0 and r==0.0 and f1==0.0
_check("zero-division edge case returns 0.0", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def precision_recall_f1(y_true, y_pred):
    # TODO
    raise NotImplementedError''',
            "solution": '''def precision_recall_f1(y_true, y_pred):
    tp = ((y_true==1)&(y_pred==1)).sum().item()
    fp = ((y_true==0)&(y_pred==1)).sum().item()
    fn = ((y_true==1)&(y_pred==0)).sum().item()
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0
    return precision, recall, f1''',
            "tests": '''
def _t1():
    y_true = torch.tensor([1,1,0,0,1]); y_pred = torch.tensor([1,0,0,0,1])
    p,r,f1 = precision_recall_f1(y_true, y_pred)
    assert abs(p-1.0)<1e-6 and abs(r-2/3)<1e-6
_check("matches hand-computed precision/recall", _t1)

def _t2():
    p,r,f1 = precision_recall_f1(torch.tensor([1,0]), torch.tensor([0,0]))
    assert p==0.0 and r==0.0 and f1==0.0
_check("zero-division edge case returns 0.0", _t2)
''',
        },
    },
    {
        "id": "dropout",
        "title": "Dropout (manual)",
        "category": "Layers & Normalization",
        "numpy": {
            "prompt": "Inverted dropout: zero w.p. p, scale survivors by 1/(1-p). Eval mode is a no-op.",
            "stub": '''def dropout_manual(x, p, training=True):
    # TODO
    raise NotImplementedError''',
            "solution": '''def dropout_manual(x, p, training=True):
    if not training or p == 0.0:
        return x
    keep_mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * keep_mask / (1 - p)''',
            "tests": '''
def _t1():
    np.random.seed(0)
    x = np.ones(20000)
    out = dropout_manual(x, p=0.5, training=True)
    zero_frac = np.mean(out == 0)
    assert 0.4 < zero_frac < 0.6
    survivors = out[out != 0]
    assert np.allclose(survivors, 2.0)
_check("~half zeroed, survivors scaled by 1/(1-p)", _t1)

def _t2():
    x = np.ones(10)
    assert np.allclose(dropout_manual(x, p=0.5, training=False), x)
_check("eval mode is a no-op", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. No nn.Dropout / F.dropout.",
            "stub": '''def dropout_manual(x, p, training=True):
    # TODO
    raise NotImplementedError''',
            "solution": '''def dropout_manual(x, p, training=True):
    if not training or p == 0.0:
        return x
    keep_mask = (torch.rand_like(x) > p).float()
    return x * keep_mask / (1.0 - p)''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    x = torch.ones(20000)
    out = dropout_manual(x, p=0.5, training=True)
    zero_frac = (out == 0).float().mean().item()
    assert 0.4 < zero_frac < 0.6
    survivors = out[out != 0]
    assert torch.allclose(survivors, torch.full_like(survivors, 2.0))
_check("~half zeroed, survivors scaled by 1/(1-p)", _t1)

def _t2():
    x = torch.ones(10)
    assert torch.allclose(dropout_manual(x, p=0.5, training=False), x)
_check("eval mode is a no-op", _t2)
''',
        },
    },
    {
        "id": "attention",
        "title": "Scaled dot-product attention",
        "category": "Transformers",
        "numpy": {
            "prompt": "softmax(QK^T / sqrt(d_k)) @ V, with optional mask (forward pass only, no gradients needed here).",
            "stub": '''def scaled_dot_product_attention(Q, K, V, mask=None):
    # TODO -- return (output, attn_weights)
    raise NotImplementedError''',
            "solution": '''def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return attn @ V, attn''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    Q = rng.randn(2,4,8); K = rng.randn(2,4,8); V = rng.randn(2,4,5)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2,4,5)
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-5)
_check("shapes correct & attn rows sum to 1", _t1)

def _t2():
    rng = np.random.RandomState(0)
    Q = rng.randn(2,4,8); K = rng.randn(2,4,8); V = rng.randn(2,4,5)
    mask = np.tril(np.ones((4,4)))
    _, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert np.allclose(attn[:,0,1:], 0, atol=1e-6)
_check("causal mask blocks attending to future positions", _t2)
''',
        },
        "torch": {
            "prompt": "Same math with torch tensors. No F.scaled_dot_product_attention.",
            "stub": '''def scaled_dot_product_attention(Q, K, V, mask=None):
    # TODO
    raise NotImplementedError''',
            "solution": '''import math
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V, attn''',
            "tests": '''
def _t1():
    Q = torch.randn(2, 4, 8); K = torch.randn(2, 4, 8); V = torch.randn(2, 4, 5)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 4, 5)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
_check("shapes correct & attn rows sum to 1", _t1)

def _t2():
    Q = torch.randn(2, 4, 8); K = torch.randn(2, 4, 8); V = torch.randn(2, 4, 5)
    mask = torch.tril(torch.ones(4, 4))
    _, attn_c = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert torch.allclose(attn_c[:, 0, 1:], torch.zeros(2, 3), atol=1e-6)
_check("causal mask blocks attending to future positions", _t2)
''',
        },
    },
    {
        "id": "multihead_attention",
        "title": "Multi-head attention (class)",
        "category": "Transformers",
        "numpy": {
            "prompt": "Unlike the others, this one is a class. Implement MultiHeadAttention.forward(x): self-attention with d_model split evenly across num_heads. Wq/Wk/Wv/Wo ((d_model,d_model)) are already created for you in __init__. Split into heads, run scaled dot-product attention per head, concatenate, then project with Wo. x:(B,T,d_model).",
            "stub": '''class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(d_model, d_model) * 0.1
        self.Wk = rng.randn(d_model, d_model) * 0.1
        self.Wv = rng.randn(d_model, d_model) * 0.1
        self.Wo = rng.randn(d_model, d_model) * 0.1

    def forward(self, x):
        # TODO
        raise NotImplementedError''',
            "solution": '''class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(d_model, d_model) * 0.1
        self.Wk = rng.randn(d_model, d_model) * 0.1
        self.Wv = rng.randn(d_model, d_model) * 0.1
        self.Wo = rng.randn(d_model, d_model) * 0.1

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv

        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        scores = Qh @ np.swapaxes(Kh, -2, -1) / np.sqrt(self.d_k)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out_heads = attn @ Vh

        out = out_heads.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ self.Wo''',
            "tests": '''
def _t1():
    mha = MultiHeadAttention(d_model=8, num_heads=2, seed=0)
    x = np.random.RandomState(3).randn(3, 5, 8)
    out = mha.forward(x)
    assert out.shape == (3, 5, 8)
_check("output shape correct", _t1)

def _t2():
    mha = MultiHeadAttention(d_model=8, num_heads=1, seed=5)
    x = np.random.RandomState(6).randn(2, 4, 8)
    out = mha.forward(x)
    Q, K, V = x @ mha.Wq, x @ mha.Wk, x @ mha.Wv
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(8)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    e = np.exp(shifted); attn = e / e.sum(axis=-1, keepdims=True)
    ref = (attn @ V) @ mha.Wo
    assert np.allclose(out, ref, atol=1e-6)
_check("num_heads=1 reduces to plain scaled dot-product attention", _t2)

def _t3():
    mha = MultiHeadAttention(d_model=8, num_heads=2, seed=1)
    x = np.random.RandomState(2).randn(2, 4, 8)
    out = mha.forward(x)
    B, T, D = x.shape
    H, dk = mha.num_heads, mha.d_k
    Q, K, V = x @ mha.Wq, x @ mha.Wk, x @ mha.Wv
    out_heads = np.zeros((B, T, H, dk))
    for h in range(H):
        sl = slice(h*dk, (h+1)*dk)
        Qh, Kh, Vh = Q[:,:,sl], K[:,:,sl], V[:,:,sl]
        sc = Qh @ np.swapaxes(Kh, -2, -1) / np.sqrt(dk)
        sh = sc - np.max(sc, axis=-1, keepdims=True)
        ee = np.exp(sh); at = ee / ee.sum(axis=-1, keepdims=True)
        out_heads[:,:,h,:] = at @ Vh
    ref = out_heads.reshape(B, T, D) @ mha.Wo
    assert np.allclose(out, ref, atol=1e-6)
_check("per-head attention matches an independent head-by-head reference", _t3)
''',
        },
        "torch": {
            "prompt": "Same, but as an nn.Module -- this is the part that's different from the rest: __init__ builds the layers (already done: self.Wq/Wk/Wv/Wo are nn.Linear(d_model, d_model, bias=False)), forward() wires them up. No nn.MultiheadAttention / F.scaled_dot_product_attention (plain torch.softmax is fine).",
            "stub": '''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # TODO
        raise NotImplementedError''',
            "solution": '''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        scores = Qh @ Kh.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out_heads = attn @ Vh

        out = out_heads.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = torch.randn(3, 5, 8)
    out = mha(x)
    assert out.shape == (3, 5, 8)
_check("output shape correct", _t1)

def _t2():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=8, num_heads=1)
    x = torch.randn(2, 4, 8)
    out = mha(x)
    Q, K, V = mha.Wq(x), mha.Wk(x), mha.Wv(x)
    scores = Q @ K.transpose(-2, -1) / (8 ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    ref = mha.Wo(attn @ V)
    assert torch.allclose(out, ref, atol=1e-5)
_check("num_heads=1 reduces to plain scaled dot-product attention", _t2)

def _t3():
    torch.manual_seed(1)
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    out = mha(x)
    B, T, D = x.shape
    H, dk = mha.num_heads, mha.d_k
    Q, K, V = mha.Wq(x), mha.Wk(x), mha.Wv(x)
    out_heads = torch.zeros(B, T, H, dk)
    for h in range(H):
        sl = slice(h*dk, (h+1)*dk)
        Qh, Kh, Vh = Q[..., sl], K[..., sl], V[..., sl]
        sc = Qh @ Kh.transpose(-2, -1) / (dk ** 0.5)
        at = torch.softmax(sc, dim=-1)
        out_heads[:, :, h, :] = at @ Vh
    ref = mha.Wo(out_heads.reshape(B, T, D))
    assert torch.allclose(out, ref, atol=1e-5)
_check("per-head attention matches an independent head-by-head reference", _t3)

def _t4():
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    n_params = sum(p.numel() for p in mha.parameters())
    assert n_params == 4 * 8 * 8
    x = torch.randn(2, 4, 8, requires_grad=True)
    out = mha(x)
    out.sum().backward()
    assert mha.Wq.weight.grad is not None and not torch.any(torch.isnan(mha.Wq.weight.grad))
_check("registers exactly the 4 projection matrices as parameters & gradients flow", _t4)
''',
        },
    },
    {
        "id": "posenc",
        "title": "Positional encoding",
        "category": "Transformers",
        "numpy": {
            "prompt": "Sinusoidal positional encoding (Transformer-style), shape (seq_len, d_model).",
            "stub": '''def positional_encoding(seq_len, d_model):
    # TODO
    raise NotImplementedError''',
            "solution": '''def positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, None].astype(float)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe''',
            "tests": '''
def _t1():
    pe = positional_encoding(10, 16)
    assert pe.shape == (10, 16)
    assert np.allclose(pe[0, 0::2], 0, atol=1e-6)
    assert np.allclose(pe[0, 1::2], 1, atol=1e-6)
_check("shape correct; sin(0)=0, cos(0)=1 at position 0", _t1)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def positional_encoding(seq_len, d_model):
    # TODO
    raise NotImplementedError''',
            "solution": '''import math
def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe''',
            "tests": '''
def _t1():
    pe = positional_encoding(10, 16)
    assert pe.shape == (10, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(8), atol=1e-6)
_check("shape correct; sin(0)=0, cos(0)=1 at position 0", _t1)
''',
        },
    },
    {
        "id": "attn_masks",
        "title": "Attention masks (causal & padding)",
        "category": "Transformers",
        "numpy": {
            "prompt": "Build the two masks every Transformer needs. causal_mask(T): (T,T) bool array, True where position i may attend to position j (j<=i). padding_mask(lengths, max_len): (B,max_len) bool array, True where position < lengths[b] (real tokens, not padding).",
            "stub": '''def causal_mask(T):
    # TODO
    raise NotImplementedError

def padding_mask(lengths, max_len):
    # TODO
    raise NotImplementedError''',
            "solution": '''def causal_mask(T):
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    return j <= i

def padding_mask(lengths, max_len):
    lengths = np.asarray(lengths)
    positions = np.arange(max_len)[None, :]
    return positions < lengths[:, None]''',
            "tests": '''
def _t1():
    m = causal_mask(4)
    expected = np.array([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert m.shape == (4, 4)
    assert np.array_equal(m, expected)
_check("causal mask matches lower-triangular pattern", _t1)

def _t2():
    m = padding_mask(np.array([2, 4, 1]), max_len=4)
    expected = np.array([
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ])
    assert m.shape == (3, 4)
    assert np.array_equal(m, expected)
_check("padding mask marks real tokens up to each sequence's length", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors (bool dtype).",
            "stub": '''def causal_mask(T):
    # TODO
    raise NotImplementedError

def padding_mask(lengths, max_len):
    # TODO
    raise NotImplementedError''',
            "solution": '''def causal_mask(T):
    i = torch.arange(T).unsqueeze(1)
    j = torch.arange(T).unsqueeze(0)
    return j <= i

def padding_mask(lengths, max_len):
    positions = torch.arange(max_len).unsqueeze(0)
    return positions < lengths.unsqueeze(1)''',
            "tests": '''
def _t1():
    m = causal_mask(4)
    expected = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert m.shape == (4, 4)
    assert torch.equal(m, expected)
_check("causal mask matches lower-triangular pattern", _t1)

def _t2():
    m = padding_mask(torch.tensor([2, 4, 1]), max_len=4)
    expected = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ])
    assert m.shape == (3, 4)
    assert torch.equal(m, expected)
_check("padding mask marks real tokens up to each sequence's length", _t2)
''',
        },
    },
    {
        "id": "ffn_sublayer",
        "title": "Position-wise feedforward sublayer",
        "category": "Transformers",
        "numpy": {
            "prompt": "Position-wise FFN: Linear(d_model->d_ff) -> ReLU -> Linear(d_ff->d_model), applied independently at every position. Weights are already created in __init__. x:(B,T,d_model).",
            "stub": '''class FeedForward:
    def __init__(self, d_model, d_ff, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # TODO
        raise NotImplementedError''',
            "solution": '''class FeedForward:
    def __init__(self, d_model, d_ff, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2''',
            "tests": '''
def _t1():
    ffn = FeedForward(d_model=6, d_ff=10, seed=0)
    x = np.random.RandomState(1).randn(2, 3, 6)
    out = ffn.forward(x)
    assert out.shape == (2, 3, 6)
    ref = np.maximum(x @ ffn.W1 + ffn.b1, 0) @ ffn.W2 + ffn.b2
    assert np.allclose(out, ref, atol=1e-6)
_check("matches Linear->ReLU->Linear applied at every position", _t1)

def _t2():
    ffn = FeedForward(d_model=4, d_ff=4, seed=2)
    ffn.W1 = np.eye(4); ffn.b1 = np.array([0.0,0.0,-10.0,0.0])
    ffn.W2 = np.eye(4); ffn.b2 = np.zeros(4)
    x = np.array([[[1.0,2.0,3.0,4.0]]])
    out = ffn.forward(x)
    assert np.allclose(out, [[[1.0,2.0,0.0,4.0]]])
_check("ReLU actually zeroes negative pre-activations", _t2)
''',
        },
        "torch": {
            "prompt": "Same, as an nn.Module. linear1/linear2 are already built for you in __init__.",
            "stub": '''class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO
        raise NotImplementedError''',
            "solution": '''class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    ffn = FeedForward(d_model=6, d_ff=10)
    x = torch.randn(2, 3, 6)
    out = ffn(x)
    ref = ffn.linear2(torch.relu(ffn.linear1(x)))
    assert out.shape == (2, 3, 6)
    assert torch.allclose(out, ref, atol=1e-6)
_check("matches Linear->ReLU->Linear applied at every position", _t1)

def _t2():
    ffn = FeedForward(d_model=6, d_ff=10)
    n_params = sum(p.numel() for p in ffn.parameters())
    assert n_params == (6*10+10) + (10*6+6)
    x = torch.randn(2, 3, 6, requires_grad=True)
    out = ffn(x)
    out.sum().backward()
    assert x.grad is not None and not torch.any(torch.isnan(x.grad))
_check("registers exactly linear1+linear2 as parameters & gradients flow", _t2)
''',
        },
    },
    {
        "id": "add_norm",
        "title": "Residual + LayerNorm sublayer (Add & Norm)",
        "category": "Transformers",
        "numpy": {
            "prompt": "The 'Add & Norm' wrapper used around every Transformer sublayer: output = LayerNorm(x + sublayer(x)), where sublayer is any callable that maps x to something the same shape. gamma/beta:(D,) are LayerNorm's own scale/shift.",
            "stub": '''def add_norm(x, sublayer, gamma, beta, eps=1e-5):
    # TODO
    raise NotImplementedError''',
            "solution": '''def add_norm(x, sublayer, gamma, beta, eps=1e-5):
    y = x + sublayer(x)
    mean = y.mean(axis=-1, keepdims=True)
    var = y.var(axis=-1, keepdims=True)
    y_hat = (y - mean) / np.sqrt(var + eps)
    return gamma * y_hat + beta''',
            "tests": '''
def _t1():
    x = np.random.RandomState(0).randn(2, 3, 4)
    gamma = np.ones(4); beta = np.zeros(4)
    out = add_norm(x, lambda t: t * 2, gamma, beta)
    y = x + x * 2
    mean = y.mean(axis=-1, keepdims=True); var = y.var(axis=-1, keepdims=True)
    ref = (y - mean) / np.sqrt(var + 1e-5)
    assert np.allclose(out, ref, atol=1e-6)
_check("applies LayerNorm to x + sublayer(x)", _t1)

def _t2():
    x = np.random.RandomState(1).randn(5, 6)
    out = add_norm(x, lambda t: np.zeros_like(t), np.ones(6), np.zeros(6))
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(out.std(axis=-1), 1, atol=1e-2)
_check("degenerates to plain LayerNorm(x) when the sublayer contributes nothing", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def add_norm(x, sublayer, gamma, beta, eps=1e-5):
    # TODO
    raise NotImplementedError''',
            "solution": '''def add_norm(x, sublayer, gamma, beta, eps=1e-5):
    y = x + sublayer(x)
    mean = y.mean(dim=-1, keepdim=True)
    var = y.var(dim=-1, unbiased=False, keepdim=True)
    y_hat = (y - mean) / torch.sqrt(var + eps)
    return gamma * y_hat + beta''',
            "tests": '''
def _t1():
    x = torch.randn(2, 3, 4)
    gamma = torch.ones(4); beta = torch.zeros(4)
    out = add_norm(x, lambda t: t * 2, gamma, beta)
    y = x + x * 2
    mean = y.mean(dim=-1, keepdim=True); var = y.var(dim=-1, unbiased=False, keepdim=True)
    ref = (y - mean) / torch.sqrt(var + 1e-5)
    assert torch.allclose(out, ref, atol=1e-5)
_check("applies LayerNorm to x + sublayer(x)", _t1)

def _t2():
    x = torch.randn(5, 6)
    out = add_norm(x, lambda t: torch.zeros_like(t), torch.ones(6), torch.zeros(6))
    assert torch.allclose(out.mean(dim=-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(5), atol=1e-2)
_check("degenerates to plain LayerNorm(x) when the sublayer contributes nothing", _t2)
''',
        },
    },
    {
        "id": "encoder_layer",
        "title": "Transformer encoder layer",
        "category": "Transformers",
        "numpy": {
            "prompt": "Wire a full Transformer encoder layer (post-norm, like the original paper): x = AddNorm(x, self_attention(x)); x = AddNorm(x, feed_forward(x)). self_attention, feed_forward and add_norm are already implemented for you in __init__/below -- forward() is just the 2-line wiring.",
            "stub": '''class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, seed=0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(d_model, d_model) * 0.1
        self.Wk = rng.randn(d_model, d_model) * 0.1
        self.Wv = rng.randn(d_model, d_model) * 0.1
        self.Wo = rng.randn(d_model, d_model) * 0.1
        self.W1 = rng.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
        self.gamma1 = np.full(d_model, 1.0); self.beta1 = np.zeros(d_model)
        self.gamma2 = np.full(d_model, 2.0); self.beta2 = np.full(d_model, 0.5)

    def self_attention(self, x):
        # already implemented for you
        B, T, D = x.shape
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv

        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scores = Qh @ np.swapaxes(Kh, -2, -1) / np.sqrt(self.d_k)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out_heads = attn @ Vh
        out = out_heads.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ self.Wo

    def feed_forward(self, x):
        # already implemented for you
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def add_norm(self, x, sublayer_out, gamma, beta, eps=1e-5):
        # already implemented for you
        y = x + sublayer_out
        mean = y.mean(axis=-1, keepdims=True)
        var = y.var(axis=-1, keepdims=True)
        y_hat = (y - mean) / np.sqrt(var + eps)
        return gamma * y_hat + beta

    def forward(self, x):
        # TODO -- x = add_norm(x, self_attention(x), gamma1, beta1)
        #         x = add_norm(x, feed_forward(x), gamma2, beta2)
        raise NotImplementedError''',
            "solution": '''class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, seed=0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(d_model, d_model) * 0.1
        self.Wk = rng.randn(d_model, d_model) * 0.1
        self.Wv = rng.randn(d_model, d_model) * 0.1
        self.Wo = rng.randn(d_model, d_model) * 0.1
        self.W1 = rng.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
        self.gamma1 = np.full(d_model, 1.0); self.beta1 = np.zeros(d_model)
        self.gamma2 = np.full(d_model, 2.0); self.beta2 = np.full(d_model, 0.5)

    def self_attention(self, x):
        B, T, D = x.shape
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv

        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scores = Qh @ np.swapaxes(Kh, -2, -1) / np.sqrt(self.d_k)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out_heads = attn @ Vh
        out = out_heads.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ self.Wo

    def feed_forward(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def add_norm(self, x, sublayer_out, gamma, beta, eps=1e-5):
        y = x + sublayer_out
        mean = y.mean(axis=-1, keepdims=True)
        var = y.var(axis=-1, keepdims=True)
        y_hat = (y - mean) / np.sqrt(var + eps)
        return gamma * y_hat + beta

    def forward(self, x):
        x = self.add_norm(x, self.self_attention(x), self.gamma1, self.beta1)
        x = self.add_norm(x, self.feed_forward(x), self.gamma2, self.beta2)
        return x''',
            "tests": '''
def _t1():
    layer = TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=16, seed=0)
    x = np.random.RandomState(1).randn(2, 5, 8)
    out = layer.forward(x)
    assert out.shape == (2, 5, 8)
_check("output shape correct", _t1)

def _t2():
    layer = TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=16, seed=0)
    x = np.random.RandomState(1).randn(2, 5, 8)
    out = layer.forward(x)
    ref = layer.add_norm(x, layer.self_attention(x), layer.gamma1, layer.beta1)
    ref = layer.add_norm(ref, layer.feed_forward(ref), layer.gamma2, layer.beta2)
    assert np.allclose(out, ref, atol=1e-6)
_check("matches self-attention -> add&norm -> feedforward -> add&norm, in that order", _t2)
''',
        },
        "torch": {
            "prompt": "Same, as an nn.Module (post-norm): x = norm1(x + self_attention(x)); x = norm2(x + feed_forward(x)). All sublayers are already built for you in __init__ -- forward() is the 2-line wiring. No nn.TransformerEncoderLayer / F.scaled_dot_product_attention.",
            "stub": '''class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        with torch.no_grad():
            self.norm2.weight.fill_(2.0)

    def self_attention(self, x):
        # already implemented for you
        B, T, D = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scores = Qh @ Kh.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out_heads = attn @ Vh
        out = out_heads.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)

    def feed_forward(self, x):
        # already implemented for you
        return self.linear2(torch.relu(self.linear1(x)))

    def forward(self, x):
        # TODO -- x = norm1(x + self_attention(x))
        #         x = norm2(x + feed_forward(x))
        raise NotImplementedError''',
            "solution": '''class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        with torch.no_grad():
            self.norm2.weight.fill_(2.0)

    def self_attention(self, x):
        B, T, D = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scores = Qh @ Kh.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out_heads = attn @ Vh
        out = out_heads.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)

    def feed_forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

    def forward(self, x):
        x = self.norm1(x + self.self_attention(x))
        x = self.norm2(x + self.feed_forward(x))
        return x''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    layer = TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=16)
    x = torch.randn(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 8)
_check("output shape correct", _t1)

def _t2():
    torch.manual_seed(0)
    layer = TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=16)
    x = torch.randn(2, 5, 8)
    out = layer(x)
    ref = layer.norm1(x + layer.self_attention(x))
    ref = layer.norm2(ref + layer.feed_forward(ref))
    assert torch.allclose(out, ref, atol=1e-5)
_check("matches self-attention -> norm1 -> feedforward -> norm2, in that order", _t2)

def _t3():
    torch.manual_seed(0)
    layer = TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=16)
    x = torch.randn(2, 5, 8, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None and not torch.any(torch.isnan(x.grad))
    assert layer.Wq.weight.grad is not None
_check("gradients flow back through both sublayers to the input and Wq", _t3)
''',
        },
    },
    {
        "id": "relu_backprop",
        "title": "ReLU forward + backward",
        "category": "Activations & Losses",
        "numpy": {
            "prompt": "Hand-roll the chain rule: given x and the upstream gradient dout, return (out, dx) for out = relu(x).",
            "stub": '''def relu_forward_backward(x, dout):
    # TODO
    raise NotImplementedError''',
            "solution": '''def relu_forward_backward(x, dout):
    out = np.where(x > 0, x, 0)
    dx = dout * (x > 0)
    return out, dx''',
            "tests": '''
def _t1():
    x = np.array([-1.0, 0.0, 2.0, 3.0])
    dout = np.array([1.0, 1.0, 1.0, 1.0])
    out, dx = relu_forward_backward(x, dout)
    assert np.allclose(out, [0,0,2,3])
    assert np.allclose(dx, [0,0,1,1])
_check("zeros negatives, passes gradient through positives", _t1)
''',
        },
        "torch": {
            "prompt": ("In numpy you hand-derive forward+backward directly. In PyTorch, autograd usually does "
                       "this for you -- here you implement the same math yourself inside a custom "
                       "torch.autograd.Function so you control both passes explicitly. "
                       "No torch.relu / F.relu / .clamp in forward."),
            "stub": '''class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        raise NotImplementedError''',
            "solution": '''class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x > 0, x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input''',
            "tests": '''
def _t1():
    x = torch.randn(20, requires_grad=True)
    out = CustomReLUFunction.apply(x)
    assert torch.allclose(out, torch.clamp(x, min=0))
    out.sum().backward()
    assert torch.allclose(x.grad, (x > 0).float())
_check("forward matches relu, backward gradient correct", _t1)
''',
        },
    },
    {
        "id": "linreg",
        "title": "Linear regression via gradient descent",
        "category": "Classical ML & Training",
        "numpy": {
            "prompt": "Fit y = Xw + b by deriving the MSE gradients yourself and updating with plain gradient descent.",
            "stub": '''def train_linear_regression(X, y, lr=0.1, epochs=200):
    # TODO
    raise NotImplementedError''',
            "solution": '''def train_linear_regression(X, y, lr=0.1, epochs=200):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        error = X @ w + b - y
        dw = (2.0/n) * (X.T @ error)
        db = (2.0/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    true_w = np.array([2.0, -3.0]); true_b = 1.0
    X = rng.randn(300, 2)
    y = X @ true_w + true_b + 0.01 * rng.randn(300)
    w, b = train_linear_regression(X, y, lr=0.1, epochs=300)
    assert np.allclose(w, true_w, atol=0.2)
    assert np.isclose(b, true_b, atol=0.2)
_check("recovers the true weights & bias", _t1)
''',
        },
        "torch": {
            "prompt": "Same optimization, but gradients come from autograd instead of being derived by hand. No torch.optim -- update manually inside torch.no_grad().",
            "stub": '''def train_linear_regression(X, y, lr=0.1, epochs=200):
    # TODO
    raise NotImplementedError''',
            "solution": '''def train_linear_regression(X, y, lr=0.1, epochs=200):
    w = torch.zeros(X.size(1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(epochs):
        pred = X @ w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_(); b.grad.zero_()
    return w.detach(), b.detach().squeeze()''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    true_w = torch.tensor([2.0, -3.0]); true_b = 1.0
    X = torch.randn(300, 2)
    y = X @ true_w + true_b + 0.01 * torch.randn(300)
    w, b = train_linear_regression(X, y, lr=0.1, epochs=300)
    assert torch.allclose(w, true_w, atol=0.2)
    assert torch.allclose(b, torch.tensor(true_b), atol=0.2)
_check("recovers the true weights & bias", _t1)
''',
        },
    },
    {
        "id": "count_params",
        "title": "Count trainable parameters",
        "category": "Misc",
        "numpy": None,
        "torch": {
            "prompt": "Total number of requires_grad=True parameters in an nn.Module. No NumPy analogue -- this is specific to framework model objects.",
            "stub": '''def count_trainable_params(model):
    # TODO
    raise NotImplementedError''',
            "solution": '''def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)''',
            "tests": '''
def _t1():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    expected = sum(p.numel() for p in model.parameters())
    assert count_trainable_params(model) == expected
_check("counts all params when all trainable", _t1)

def _t2():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    expected = sum(p.numel() for p in model.parameters())
    model[0].weight.requires_grad = False
    assert count_trainable_params(model) == expected - model[0].weight.numel()
_check("excludes frozen params", _t2)
''',
        },
    },
    {
        "id": "discounted_return",
        "title": "Discounted return (return-to-go)",
        "category": "RL Foundations",
        "numpy": {
            "prompt": "Given the per-step rewards of a full episode, compute the discounted return-to-go at every timestep: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... Return an array the same length as rewards.",
            "stub": '''def discounted_returns(rewards, gamma):
    # TODO
    raise NotImplementedError''',
            "solution": '''def discounted_returns(rewards, gamma):
    T = len(rewards)
    returns = np.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns''',
            "tests": '''
def _t1():
    r = discounted_returns([1, 1, 1], 0.5)
    assert np.allclose(r, [1.75, 1.5, 1.0])
_check("matches hand-computed returns-to-go", _t1)

def _t2():
    rng = np.random.RandomState(0)
    rewards = rng.randn(6)
    gamma = 0.9
    r = discounted_returns(rewards, gamma)
    # G_t should equal r_t + gamma * G_{t+1}
    for t in range(5):
        assert np.isclose(r[t], rewards[t] + gamma * r[t+1], atol=1e-8)
    assert np.isclose(r[-1], rewards[-1], atol=1e-8)
_check("satisfies the recursive relation G_t = r_t + gamma*G_(t+1)", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def discounted_returns(rewards, gamma):
    # TODO
    raise NotImplementedError''',
            "solution": '''def discounted_returns(rewards, gamma):
    T = len(rewards)
    returns = torch.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns''',
            "tests": '''
def _t1():
    r = discounted_returns(torch.tensor([1.,1.,1.]), 0.5)
    assert torch.allclose(r, torch.tensor([1.75,1.5,1.0]))
_check("matches hand-computed returns-to-go", _t1)

def _t2():
    torch.manual_seed(0)
    rewards = torch.randn(6)
    gamma = 0.9
    r = discounted_returns(rewards, gamma)
    for t in range(5):
        assert torch.isclose(r[t], rewards[t] + gamma * r[t+1], atol=1e-6)
    assert torch.isclose(r[-1], rewards[-1], atol=1e-6)
_check("satisfies the recursive relation G_t = r_t + gamma*G_(t+1)", _t2)
''',
        },
    },
    {
        "id": "epsilon_greedy",
        "title": "Epsilon-greedy action selection",
        "category": "RL Foundations",
        "numpy": {
            "prompt": "With probability epsilon, pick a uniformly random action; otherwise pick argmax(q_values). `rng` is a stdlib random.Random instance -- use rng.random() for the epsilon roll and rng.randrange(n) for the random action, so behavior is reproducible.",
            "stub": '''def epsilon_greedy_action(q_values, epsilon, rng):
    # TODO
    raise NotImplementedError''',
            "solution": '''def epsilon_greedy_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(len(q_values))
    return int(np.argmax(q_values))''',
            "tests": '''
import random

def _t1():
    q = np.array([1.0, 5.0, 2.0])
    rng = random.Random(0)
    for _ in range(20):
        assert epsilon_greedy_action(q, 0.0, rng) == 1
_check("epsilon=0 always picks the greedy action", _t1)

def _t2():
    q = np.array([1.0, 5.0, 2.0])
    rng = random.Random(0)
    counts = [0, 0, 0]
    for _ in range(3000):
        counts[epsilon_greedy_action(q, 1.0, rng)] += 1
    fracs = [c / 3000 for c in counts]
    assert all(0.2 < f < 0.45 for f in fracs)
_check("epsilon=1 explores roughly uniformly over actions", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with a torch tensor of Q-values.",
            "stub": '''def epsilon_greedy_action(q_values, epsilon, rng):
    # TODO
    raise NotImplementedError''',
            "solution": '''def epsilon_greedy_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(len(q_values))
    return int(torch.argmax(q_values).item())''',
            "tests": '''
import random

def _t1():
    q = torch.tensor([1.0, 5.0, 2.0])
    rng = random.Random(0)
    for _ in range(20):
        assert epsilon_greedy_action(q, 0.0, rng) == 1
_check("epsilon=0 always picks the greedy action", _t1)

def _t2():
    q = torch.tensor([1.0, 5.0, 2.0])
    rng = random.Random(0)
    counts = [0, 0, 0]
    for _ in range(3000):
        counts[epsilon_greedy_action(q, 1.0, rng)] += 1
    fracs = [c / 3000 for c in counts]
    assert all(0.2 < f < 0.45 for f in fracs)
_check("epsilon=1 explores roughly uniformly over actions", _t2)
''',
        },
    },
    {
        "id": "gae",
        "title": "Generalized Advantage Estimation (GAE)",
        "category": "RL Foundations",
        "numpy": {
            "prompt": "GAE-lambda: rewards (T,), values (T+1,) (includes the bootstrap value V(s_T)), dones (T,) (1 if the episode ended after step t -- masks out bootstrapping across episode boundaries). Return the (T,) advantage estimates.",
            "stub": '''def gae(rewards, values, dones, gamma, lam):
    # TODO
    raise NotImplementedError''',
            "solution": '''def gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    return advantages''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    T = 6
    rewards = rng.randn(T); values = rng.randn(T+1); dones = np.zeros(T)
    gamma, lam = 0.9, 0.95
    adv = gae(rewards, values, dones, gamma, lam)
    assert adv.shape == (T,)
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    ref = np.zeros(T)
    for t in range(T):
        s, coef = 0.0, 1.0
        for k in range(t, T):
            s += coef * deltas[k]
            coef *= gamma * lam
        ref[t] = s
    assert np.allclose(adv, ref, atol=1e-6)
_check("matches the direct sum-of-discounted-TD-errors definition", _t1)

def _t2():
    rewards = np.array([1.0, 1.0])
    values = np.array([0.0, 0.0, 100.0])  # huge bootstrap that must not leak across the episode boundary
    dones = np.array([1.0, 0.0])
    adv = gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert np.isclose(adv[0], 1.0, atol=1e-6)
_check("a done flag masks both the bootstrap value and the propagated future advantage", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def gae(rewards, values, dones, gamma, lam):
    # TODO
    raise NotImplementedError''',
            "solution": '''def gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    return advantages''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    T = 6
    rewards = torch.randn(T); values = torch.randn(T+1); dones = torch.zeros(T)
    gamma, lam = 0.9, 0.95
    adv = gae(rewards, values, dones, gamma, lam)
    assert adv.shape == (T,)
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    ref = torch.zeros(T)
    for t in range(T):
        s, coef = 0.0, 1.0
        for k in range(t, T):
            s += coef * deltas[k].item()
            coef *= gamma * lam
        ref[t] = s
    assert torch.allclose(adv, ref, atol=1e-5)
_check("matches the direct sum-of-discounted-TD-errors definition", _t1)

def _t2():
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.0, 0.0, 100.0])
    dones = torch.tensor([1.0, 0.0])
    adv = gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert torch.isclose(adv[0], torch.tensor(1.0), atol=1e-6)
_check("a done flag masks both the bootstrap value and the propagated future advantage", _t2)
''',
        },
    },
    {
        "id": "q_learning_update",
        "title": "Tabular Q-learning update",
        "category": "Value-Based RL",
        "numpy": {
            "prompt": "One Bellman TD update for tabular Q-learning: Q[s,a] += alpha*(r + gamma*max(Q[s']) * (1-done) - Q[s,a]). Return the updated table (don't mutate the input).",
            "stub": '''def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    # TODO
    raise NotImplementedError''',
            "solution": '''def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    Q = Q.copy()
    target = reward + gamma * np.max(Q[next_state]) * (1 - done)
    Q[state, action] += alpha * (target - Q[state, action])
    return Q''',
            "tests": '''
def _t1():
    Q = np.zeros((3, 2)); Q[1] = [0.5, 0.2]
    Q_new = q_learning_update(Q, state=0, action=1, reward=1.0, next_state=1, done=0, alpha=0.1, gamma=0.9)
    target = 1.0 + 0.9 * 0.5
    expected = 0.1 * (target - 0.0)
    assert np.isclose(Q_new[0,1], expected, atol=1e-8)
    assert np.allclose(Q_new[1], Q[1])
_check("matches the Bellman TD update; doesn't touch other rows", _t1)

def _t2():
    Q = np.zeros((3, 2)); Q[1] = [0.5, 0.2]
    Q_new = q_learning_update(Q, state=0, action=1, reward=1.0, next_state=1, done=1, alpha=0.1, gamma=0.9)
    assert np.isclose(Q_new[0,1], 0.1 * 1.0, atol=1e-8)
_check("done=1 removes the bootstrapped next-state value", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with a torch tensor Q-table.",
            "stub": '''def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    # TODO
    raise NotImplementedError''',
            "solution": '''def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    Q = Q.clone()
    target = reward + gamma * torch.max(Q[next_state]) * (1 - done)
    Q[state, action] += alpha * (target - Q[state, action])
    return Q''',
            "tests": '''
def _t1():
    Q = torch.zeros(3, 2); Q[1] = torch.tensor([0.5, 0.2])
    Q_new = q_learning_update(Q, state=0, action=1, reward=1.0, next_state=1, done=0, alpha=0.1, gamma=0.9)
    target = 1.0 + 0.9 * 0.5
    expected = 0.1 * (target - 0.0)
    assert torch.isclose(Q_new[0,1], torch.tensor(expected), atol=1e-6)
    assert torch.allclose(Q_new[1], Q[1])
_check("matches the Bellman TD update; doesn't touch other rows", _t1)

def _t2():
    Q = torch.zeros(3, 2); Q[1] = torch.tensor([0.5, 0.2])
    Q_new = q_learning_update(Q, state=0, action=1, reward=1.0, next_state=1, done=1, alpha=0.1, gamma=0.9)
    assert torch.isclose(Q_new[0,1], torch.tensor(0.1 * 1.0), atol=1e-6)
_check("done=1 removes the bootstrapped next-state value", _t2)
''',
        },
    },
    {
        "id": "dqn_loss",
        "title": "DQN loss (TD target + target network)",
        "category": "Value-Based RL",
        "numpy": None,
        "torch": {
            "prompt": "The DQN loss: MSE between Q(s,a) and a TD target built from a separate target network. The target must not contribute gradients (torch.no_grad() / detach) -- that's the whole point of having a target network.",
            "stub": '''def dqn_loss(q_net, target_net, states, actions, rewards, next_states, dones, gamma=0.99):
    # TODO
    raise NotImplementedError''',
            "solution": '''def dqn_loss(q_net, target_net, states, actions, rewards, next_states, dones, gamma=0.99):
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        target = rewards + gamma * next_q * (1 - dones)
    return torch.nn.functional.mse_loss(q_values, target)''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    q_net = nn.Linear(4, 3); target_net = nn.Linear(4, 3)
    states = torch.randn(5, 4); actions = torch.randint(0, 3, (5,))
    rewards = torch.randn(5); next_states = torch.randn(5, 4); dones = torch.zeros(5)
    loss = dqn_loss(q_net, target_net, states, actions, rewards, next_states, dones)
    assert loss.dim() == 0
    with torch.no_grad():
        q_ref = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_ref = target_net(next_states).max(dim=1).values
        target_ref = rewards + 0.99 * next_q_ref * (1 - dones)
        ref_loss = torch.nn.functional.mse_loss(q_ref, target_ref)
    assert torch.isclose(loss, ref_loss, atol=1e-5)
_check("matches the TD-target MSE loss", _t1)

def _t2():
    torch.manual_seed(0)
    q_net = nn.Linear(4, 3); target_net = nn.Linear(4, 3)
    states = torch.randn(5, 4); actions = torch.randint(0, 3, (5,))
    rewards = torch.randn(5); next_states = torch.randn(5, 4); dones = torch.zeros(5)
    loss = dqn_loss(q_net, target_net, states, actions, rewards, next_states, dones)
    loss.backward()
    assert q_net.weight.grad is not None and not torch.any(torch.isnan(q_net.weight.grad))
    assert target_net.weight.grad is None
_check("gradients flow into q_net but the target network stays detached", _t2)
''',
        },
    },
    {
        "id": "double_dqn_target",
        "title": "Double DQN target",
        "category": "Value-Based RL",
        "numpy": None,
        "torch": {
            "prompt": "Double DQN's fix for Q-learning's overestimation bias: use the ONLINE network to pick the best next action, but the TARGET network to evaluate its value. (Plain DQN uses the target network for both.)",
            "stub": '''def double_dqn_target(q_net, target_net, next_states, rewards, dones, gamma=0.99):
    # TODO
    raise NotImplementedError''',
            "solution": '''def double_dqn_target(q_net, target_net, next_states, rewards, dones, gamma=0.99):
    with torch.no_grad():
        best_actions = q_net(next_states).argmax(dim=1)
        next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * next_q * (1 - dones)
    return target''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    q_net = nn.Linear(4, 3); target_net = nn.Linear(4, 3)
    next_states = torch.randn(6, 4); rewards = torch.randn(6); dones = torch.zeros(6)
    target = double_dqn_target(q_net, target_net, next_states, rewards, dones, gamma=0.99)
    with torch.no_grad():
        best_actions_ref = q_net(next_states).argmax(dim=1)
        next_q_ref = target_net(next_states).gather(1, best_actions_ref.unsqueeze(1)).squeeze(1)
        ref = rewards + 0.99 * next_q_ref * (1 - dones)
    assert torch.allclose(target, ref, atol=1e-5)
_check("selects the action with the online net, evaluates it with the target net", _t1)

def _t2():
    torch.manual_seed(3)
    q_net = nn.Linear(4, 3); target_net = nn.Linear(4, 3)
    next_states = torch.randn(20, 4); rewards = torch.zeros(20); dones = torch.zeros(20)
    double_target = double_dqn_target(q_net, target_net, next_states, rewards, dones, gamma=0.99)
    with torch.no_grad():
        vanilla_target = rewards + 0.99 * target_net(next_states).max(dim=1).values * (1 - dones)
    assert not torch.allclose(double_target, vanilla_target, atol=1e-6)
_check("differs from vanilla DQN's target when the two networks disagree on the best action", _t2)
''',
        },
    },
    {
        "id": "reinforce_loss",
        "title": "REINFORCE (vanilla policy gradient) loss",
        "category": "Policy-Based RL",
        "numpy": {
            "prompt": "The REINFORCE loss for a batch of sampled actions: -mean(log_prob(a_t) * G_t). Minimizing this loss performs gradient ASCENT on expected return.",
            "stub": '''def reinforce_loss(log_probs, returns):
    # TODO
    raise NotImplementedError''',
            "solution": '''def reinforce_loss(log_probs, returns):
    return -(log_probs * returns).mean()''',
            "tests": '''
def _t1():
    log_probs = np.array([-0.5, -1.0, -0.2])
    returns = np.array([2.0, 1.0, 3.0])
    loss = reinforce_loss(log_probs, returns)
    assert np.isclose(loss, -np.mean(log_probs * returns), atol=1e-8)
_check("matches -mean(log_prob * return)", _t1)

def _t2():
    log_probs = np.array([-1.0, -1.0])
    returns = np.array([5.0, 5.0])
    assert reinforce_loss(log_probs, returns) > 0  # -(-1*5) averaged = 5 > 0
_check("penalizes low log-prob on high-return actions (loss should be positive here)", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors and gradients flowing through log_probs.",
            "stub": '''def reinforce_loss(log_probs, returns):
    # TODO
    raise NotImplementedError''',
            "solution": '''def reinforce_loss(log_probs, returns):
    return -(log_probs * returns).mean()''',
            "tests": '''
def _t1():
    log_probs = torch.tensor([-0.5, -1.0, -0.2])
    returns = torch.tensor([2.0, 1.0, 3.0])
    loss = reinforce_loss(log_probs, returns)
    assert torch.isclose(loss, -(log_probs * returns).mean(), atol=1e-6)
_check("matches -mean(log_prob * return)", _t1)

def _t2():
    log_probs = torch.tensor([-0.5, -1.0, -0.2], requires_grad=True)
    returns = torch.tensor([2.0, 1.0, 3.0])
    loss = reinforce_loss(log_probs, returns)
    loss.backward()
    assert log_probs.grad is not None
    assert torch.allclose(log_probs.grad, -returns / 3, atol=1e-6)
_check("gradient w.r.t. log_probs is -returns/N", _t2)
''',
        },
    },
    {
        "id": "ppo_clip_loss",
        "title": "PPO clipped surrogate objective",
        "category": "Policy-Based RL",
        "numpy": {
            "prompt": "PPO's clipped objective: ratio = exp(new_log_prob - old_log_prob); loss = -mean(min(ratio*advantage, clip(ratio, 1-eps, 1+eps)*advantage)).",
            "stub": '''def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    # TODO
    raise NotImplementedError''',
            "solution": '''def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    ratio = np.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.mean(np.minimum(unclipped, clipped))''',
            "tests": '''
def _t1():
    old_lp = np.array([-1.0, -1.0, -1.0, -1.0])
    new_lp = np.array([-1.0, -0.5, -1.5, -0.3])
    adv = np.array([1.0, 1.0, -1.0, -1.0])
    loss = ppo_clip_loss(old_lp, new_lp, adv, clip_eps=0.2)
    ratio = np.exp(new_lp - old_lp)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 0.8, 1.2) * adv
    expected = -np.mean(np.minimum(unclipped, clipped))
    assert np.isclose(loss, expected, atol=1e-6)
_check("matches the clipped surrogate objective", _t1)

def _t2():
    old_lp = np.array([-1.0, -2.0, 0.5])
    new_lp = old_lp.copy()
    adv = np.array([2.0, -1.0, 0.5])
    loss = ppo_clip_loss(old_lp, new_lp, adv, clip_eps=0.2)
    assert np.isclose(loss, -np.mean(adv), atol=1e-6)
_check("reduces to -mean(advantage) when the policy hasn't changed (ratio=1)", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors and gradients flowing through new_log_probs.",
            "stub": '''def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    # TODO
    raise NotImplementedError''',
            "solution": '''def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.minimum(unclipped, clipped).mean()''',
            "tests": '''
def _t1():
    old_lp = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    new_lp = torch.tensor([-1.0, -0.5, -1.5, -0.3])
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = ppo_clip_loss(old_lp, new_lp, adv, clip_eps=0.2)
    ratio = torch.exp(new_lp - old_lp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 0.8, 1.2) * adv
    expected = -torch.minimum(unclipped, clipped).mean()
    assert torch.isclose(loss, expected, atol=1e-5)
_check("matches the clipped surrogate objective", _t1)

def _t2():
    old_lp = torch.tensor([-1.0, -2.0, 0.5])
    new_lp = torch.tensor([-1.0, -2.0, 0.5], requires_grad=True)
    adv = torch.tensor([2.0, -1.0, 0.5])
    loss = ppo_clip_loss(old_lp, new_lp, adv, clip_eps=0.2)
    assert torch.isclose(loss, -adv.mean(), atol=1e-5)
    loss.backward()
    assert new_lp.grad is not None
_check("reduces to -mean(advantage) at ratio=1, and gradients flow", _t2)
''',
        },
    },
    {
        "id": "actor_critic_loss",
        "title": "Actor-critic combined loss",
        "category": "Policy-Based RL",
        "numpy": {
            "prompt": "The standard A2C-style combined loss: policy loss (REINFORCE-style, using advantages instead of raw returns) + c1*value_loss (MSE) - c2*entropy bonus (to encourage exploration).",
            "stub": '''def actor_critic_loss(log_probs, advantages, values, returns, entropy, c1=0.5, c2=0.01):
    # TODO
    raise NotImplementedError''',
            "solution": '''def actor_critic_loss(log_probs, advantages, values, returns, entropy, c1=0.5, c2=0.01):
    policy_loss = -(log_probs * advantages).mean()
    value_loss = ((values - returns) ** 2).mean()
    return policy_loss + c1 * value_loss - c2 * entropy.mean()''',
            "tests": '''
def _t1():
    lp = np.array([-0.5, -1.2]); adv = np.array([1.0, -0.5])
    v = np.array([0.5, 0.2]); ret = np.array([1.0, 0.0]); ent = np.array([0.1, 0.2])
    loss = actor_critic_loss(lp, adv, v, ret, ent, c1=0.5, c2=0.01)
    expected = -(lp * adv).mean() + 0.5 * ((v - ret) ** 2).mean() - 0.01 * ent.mean()
    assert np.isclose(loss, expected, atol=1e-8)
_check("matches policy_loss + c1*value_loss - c2*entropy", _t1)

def _t2():
    lp = np.array([-1.0]); adv = np.array([0.0]); v = np.array([3.0]); ret = np.array([3.0]); ent = np.array([0.0])
    loss = actor_critic_loss(lp, adv, v, ret, ent, c1=0.5, c2=0.01)
    assert np.isclose(loss, 0.0, atol=1e-8)  # zero advantage, zero value error, zero entropy -> zero loss
_check("degenerates to 0 when advantage, value error, and entropy are all 0", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def actor_critic_loss(log_probs, advantages, values, returns, entropy, c1=0.5, c2=0.01):
    # TODO
    raise NotImplementedError''',
            "solution": '''def actor_critic_loss(log_probs, advantages, values, returns, entropy, c1=0.5, c2=0.01):
    policy_loss = -(log_probs * advantages).mean()
    value_loss = ((values - returns) ** 2).mean()
    return policy_loss + c1 * value_loss - c2 * entropy.mean()''',
            "tests": '''
def _t1():
    lp = torch.tensor([-0.5, -1.2]); adv = torch.tensor([1.0, -0.5])
    v = torch.tensor([0.5, 0.2]); ret = torch.tensor([1.0, 0.0]); ent = torch.tensor([0.1, 0.2])
    loss = actor_critic_loss(lp, adv, v, ret, ent, c1=0.5, c2=0.01)
    expected = -(lp * adv).mean() + 0.5 * ((v - ret) ** 2).mean() - 0.01 * ent.mean()
    assert torch.isclose(loss, expected, atol=1e-6)
_check("matches policy_loss + c1*value_loss - c2*entropy", _t1)

def _t2():
    lp = torch.tensor([-0.5], requires_grad=True)
    v = torch.tensor([0.5], requires_grad=True)
    adv = torch.tensor([1.0]); ret = torch.tensor([1.0]); ent = torch.tensor([0.1])
    loss = actor_critic_loss(lp, adv, v, ret, ent)
    loss.backward()
    assert lp.grad is not None and v.grad is not None
_check("gradients flow to both log_probs and values", _t2)
''',
        },
    },
    {
        "id": "kl_penalty",
        "title": "KL penalty estimator (RLHF-style)",
        "category": "RL for LLMs (RLHF)",
        "numpy": {
            "prompt": "The low-variance, always-non-negative KL estimator used in PPO/GRPO-style RLHF (Schulman's 'k3' estimator): given log_ratio = log(pi/pi_ref) per token, KL ~= exp(log_ratio) - 1 - log_ratio.",
            "stub": '''def kl_penalty_k3(log_ratio):
    # TODO
    raise NotImplementedError''',
            "solution": '''def kl_penalty_k3(log_ratio):
    return np.exp(log_ratio) - 1 - log_ratio''',
            "tests": '''
def _t1():
    log_ratio = np.array([0.0, 0.5, -0.5, 2.0, -2.0])
    kl = kl_penalty_k3(log_ratio)
    assert np.allclose(kl, np.exp(log_ratio) - 1 - log_ratio, atol=1e-8)
    assert np.isclose(kl[0], 0.0, atol=1e-8)
_check("matches exp(log_ratio) - 1 - log_ratio; 0 when the policies agree", _t1)

def _t2():
    rng = np.random.RandomState(0)
    log_ratio = rng.randn(1000) * 2
    kl = kl_penalty_k3(log_ratio)
    assert np.all(kl >= -1e-8)
_check("always non-negative, for any log_ratio", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def kl_penalty_k3(log_ratio):
    # TODO
    raise NotImplementedError''',
            "solution": '''def kl_penalty_k3(log_ratio):
    return torch.exp(log_ratio) - 1 - log_ratio''',
            "tests": '''
def _t1():
    log_ratio = torch.tensor([0.0, 0.5, -0.5, 2.0, -2.0])
    kl = kl_penalty_k3(log_ratio)
    assert torch.allclose(kl, torch.exp(log_ratio) - 1 - log_ratio, atol=1e-6)
    assert torch.isclose(kl[0], torch.tensor(0.0), atol=1e-6)
_check("matches exp(log_ratio) - 1 - log_ratio; 0 when the policies agree", _t1)

def _t2():
    torch.manual_seed(0)
    log_ratio = torch.randn(1000) * 2
    kl = kl_penalty_k3(log_ratio)
    assert torch.all(kl >= -1e-6)
_check("always non-negative, for any log_ratio", _t2)
''',
        },
    },
    {
        "id": "grpo_advantage",
        "title": "GRPO group-relative advantage",
        "category": "RL for LLMs (RLHF)",
        "numpy": {
            "prompt": "GRPO's advantage estimate (no value network needed): sample a group of G completions per prompt, then normalize each completion's reward against its own group's mean/std. rewards:(num_groups, group_size).",
            "stub": '''def grpo_advantage(rewards, eps=1e-8):
    # TODO
    raise NotImplementedError''',
            "solution": '''def grpo_advantage(rewards, eps=1e-8):
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + eps)''',
            "tests": '''
def _t1():
    rewards = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    adv = grpo_advantage(rewards)
    expected0 = (rewards[0] - rewards[0].mean()) / (rewards[0].std() + 1e-8)
    assert adv.shape == (2, 3)
    assert np.allclose(adv[0], expected0, atol=1e-6)
    assert np.allclose(adv[1], 0.0, atol=1e-4)
_check("matches per-group z-score; a zero-variance group stays finite (not NaN/inf)", _t1)

def _t2():
    rng = np.random.RandomState(0)
    rewards = rng.randn(4, 8) * 3 + 5
    adv = grpo_advantage(rewards)
    assert np.allclose(adv.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(adv.std(axis=1), 1, atol=1e-2)
_check("each group is normalized to mean 0 / std 1 independently", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. Use unbiased=False for std to match the population-std convention used everywhere else here.",
            "stub": '''def grpo_advantage(rewards, eps=1e-8):
    # TODO
    raise NotImplementedError''',
            "solution": '''def grpo_advantage(rewards, eps=1e-8):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)''',
            "tests": '''
def _t1():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    adv = grpo_advantage(rewards)
    mean0 = rewards[0].mean(); std0 = rewards[0].std(unbiased=False)
    expected0 = (rewards[0] - mean0) / (std0 + 1e-8)
    assert adv.shape == (2, 3)
    assert torch.allclose(adv[0], expected0, atol=1e-5)
    assert torch.allclose(adv[1], torch.zeros(3), atol=1e-4)
_check("matches per-group z-score; a zero-variance group stays finite (not NaN/inf)", _t1)

def _t2():
    torch.manual_seed(0)
    rewards = torch.randn(4, 8) * 3 + 5
    adv = grpo_advantage(rewards)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(adv.std(dim=1, unbiased=False), torch.ones(4), atol=1e-2)
_check("each group is normalized to mean 0 / std 1 independently", _t2)
''',
        },
    },
    {
        "id": "grpo_loss",
        "title": "GRPO loss (clip + KL, no value network)",
        "category": "RL for LLMs (RLHF)",
        "numpy": {
            "prompt": "The full GRPO objective: PPO's clipped surrogate, but with group-relative advantages instead of a learned value function, minus a KL penalty against a reference policy. old/new/ref_log_probs and rewards all have shape (num_groups, group_size).",
            "stub": '''def grpo_loss(old_log_probs, new_log_probs, ref_log_probs, rewards, clip_eps=0.2, kl_coef=0.04):
    # TODO
    raise NotImplementedError''',
            "solution": '''def grpo_loss(old_log_probs, new_log_probs, ref_log_probs, rewards, clip_eps=0.2, kl_coef=0.04):
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    advantages = (rewards - mean) / (std + 1e-8)

    ratio = np.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_term = np.minimum(unclipped, clipped)

    log_ratio_kl = new_log_probs - ref_log_probs
    kl = np.exp(log_ratio_kl) - 1 - log_ratio_kl

    return -np.mean(policy_term - kl_coef * kl)''',
            "tests": '''
def _t1():
    rng = np.random.RandomState(0)
    G, N = 3, 4
    rewards = rng.randn(G, N)
    old_lp = rng.randn(G, N) * 0.1 - 1.0
    new_lp = old_lp + rng.randn(G, N) * 0.1
    ref_lp = old_lp + rng.randn(G, N) * 0.05
    loss = grpo_loss(old_lp, new_lp, ref_lp, rewards, clip_eps=0.2, kl_coef=0.04)

    mean = rewards.mean(axis=1, keepdims=True); std = rewards.std(axis=1, keepdims=True)
    adv = (rewards - mean) / (std + 1e-8)
    ratio = np.exp(new_lp - old_lp)
    unclipped = ratio * adv; clipped = np.clip(ratio, 0.8, 1.2) * adv
    policy_term = np.minimum(unclipped, clipped)
    log_ratio_kl = new_lp - ref_lp
    kl = np.exp(log_ratio_kl) - 1 - log_ratio_kl
    expected = -np.mean(policy_term - 0.04 * kl)
    assert np.isclose(loss, expected, atol=1e-6)
_check("matches group-normalized PPO-clip objective minus a KL penalty vs. the reference policy", _t1)

def _t2():
    rng = np.random.RandomState(1)
    rewards = rng.randn(2, 5)
    lp = rng.randn(2, 5)
    loss = grpo_loss(lp, lp, lp, rewards)
    mean = rewards.mean(axis=1, keepdims=True); std = rewards.std(axis=1, keepdims=True)
    adv = (rewards - mean) / (std + 1e-8)
    assert np.isclose(loss, -np.mean(adv), atol=1e-6)
_check("reduces to -mean(advantage) when new==old==ref (no policy change, no KL)", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors.",
            "stub": '''def grpo_loss(old_log_probs, new_log_probs, ref_log_probs, rewards, clip_eps=0.2, kl_coef=0.04):
    # TODO
    raise NotImplementedError''',
            "solution": '''def grpo_loss(old_log_probs, new_log_probs, ref_log_probs, rewards, clip_eps=0.2, kl_coef=0.04):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    advantages = (rewards - mean) / (std + 1e-8)

    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_term = torch.minimum(unclipped, clipped)

    log_ratio_kl = new_log_probs - ref_log_probs
    kl = torch.exp(log_ratio_kl) - 1 - log_ratio_kl

    return -(policy_term - kl_coef * kl).mean()''',
            "tests": '''
def _t1():
    torch.manual_seed(0)
    G, N = 3, 4
    rewards = torch.randn(G, N)
    old_lp = torch.randn(G, N) * 0.1 - 1.0
    new_lp = old_lp + torch.randn(G, N) * 0.1
    ref_lp = old_lp + torch.randn(G, N) * 0.05
    loss = grpo_loss(old_lp, new_lp, ref_lp, rewards, clip_eps=0.2, kl_coef=0.04)

    mean = rewards.mean(dim=1, keepdim=True); std = rewards.std(dim=1, unbiased=False, keepdim=True)
    adv = (rewards - mean) / (std + 1e-8)
    ratio = torch.exp(new_lp - old_lp)
    unclipped = ratio * adv; clipped = torch.clamp(ratio, 0.8, 1.2) * adv
    policy_term = torch.minimum(unclipped, clipped)
    log_ratio_kl = new_lp - ref_lp
    kl = torch.exp(log_ratio_kl) - 1 - log_ratio_kl
    expected = -(policy_term - 0.04 * kl).mean()
    assert torch.isclose(loss, expected, atol=1e-5)
_check("matches group-normalized PPO-clip objective minus a KL penalty vs. the reference policy", _t1)

def _t2():
    torch.manual_seed(1)
    rewards = torch.randn(2, 5)
    lp = torch.randn(2, 5)
    loss = grpo_loss(lp, lp, lp, rewards)
    mean = rewards.mean(dim=1, keepdim=True); std = rewards.std(dim=1, unbiased=False, keepdim=True)
    adv = (rewards - mean) / (std + 1e-8)
    assert torch.isclose(loss, -adv.mean(), atol=1e-5)
_check("reduces to -mean(advantage) when new==old==ref (no policy change, no KL)", _t2)
''',
        },
    },
    {
        "id": "reward_model_loss",
        "title": "Reward model loss (Bradley-Terry)",
        "category": "RL for LLMs (RLHF)",
        "numpy": {
            "prompt": "The reward model training loss in the classic RLHF pipeline: given a pair of scalar reward scores for a (chosen, rejected) completion of the same prompt, the Bradley-Terry pairwise loss is -log(sigmoid(r_chosen - r_rejected)). Implement it in the numerically-stable log-sum-exp form (no explicit sigmoid call).",
            "stub": '''def reward_model_loss(chosen_rewards, rejected_rewards):
    # TODO
    raise NotImplementedError''',
            "solution": '''def reward_model_loss(chosen_rewards, rejected_rewards):
    diff = chosen_rewards - rejected_rewards
    return np.mean(np.logaddexp(0, -diff))''',
            "tests": '''
def _t1():
    chosen = np.array([2.0, 1.0, 0.5])
    rejected = np.array([1.0, 1.5, 0.5])
    loss = reward_model_loss(chosen, rejected)
    diff = chosen - rejected
    expected = np.mean(-np.log(1 / (1 + np.exp(-diff))))
    assert np.isclose(loss, expected, atol=1e-6)
_check("matches -log(sigmoid(chosen - rejected))", _t1)

def _t2():
    loss = reward_model_loss(np.array([1000.0]), np.array([-1000.0]))
    assert np.isfinite(loss) and loss < 1e-6
_check("stable and near-zero when the model confidently ranks chosen above rejected", _t2)

def _t3():
    loss = reward_model_loss(np.array([-1000.0]), np.array([1000.0]))
    assert np.isfinite(loss) and loss > 100
_check("stable (not inf/nan) even when confidently WRONG", _t3)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors. Use F.logsigmoid rather than composing log+sigmoid yourself, for the same numerical-stability reason.",
            "stub": '''def reward_model_loss(chosen_rewards, rejected_rewards):
    # TODO
    raise NotImplementedError''',
            "solution": '''def reward_model_loss(chosen_rewards, rejected_rewards):
    diff = chosen_rewards - rejected_rewards
    return -torch.nn.functional.logsigmoid(diff).mean()''',
            "tests": '''
def _t1():
    chosen = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    rejected = torch.tensor([1.0, 1.5, 0.5])
    loss = reward_model_loss(chosen, rejected)
    diff = chosen.detach() - rejected
    expected = torch.mean(-torch.log(1 / (1 + torch.exp(-diff))))
    assert torch.isclose(loss, expected, atol=1e-5)
_check("matches -log(sigmoid(chosen - rejected))", _t1)

def _t2():
    chosen = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    rejected = torch.tensor([1.0, 1.5, 0.5])
    loss = reward_model_loss(chosen, rejected)
    loss.backward()
    assert chosen.grad is not None and not torch.any(torch.isnan(chosen.grad))
_check("gradient flows back to the reward scores", _t2)
''',
        },
    },
    {
        "id": "dpo_loss",
        "title": "DPO (Direct Preference Optimization) loss",
        "category": "RL for LLMs (RLHF)",
        "numpy": {
            "prompt": "DPO reformulates RLHF as a direct classification loss over preference pairs -- no reward model, no RL loop. Given log-probs (summed over the response) of chosen/rejected completions under the policy and a frozen reference model: logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)); loss = -log(sigmoid(logits)), in stable log-sum-exp form.",
            "stub": '''def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # TODO
    raise NotImplementedError''',
            "solution": '''def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    return np.mean(np.logaddexp(0, -logits))''',
            "tests": '''
def _t1():
    pc = np.array([-1.0, -0.5]); pr = np.array([-2.0, -1.5])
    rc = np.array([-1.2, -0.6]); rr = np.array([-1.8, -1.4])
    loss = dpo_loss(pc, pr, rc, rr, beta=0.1)
    policy_logratios = pc - pr; ref_logratios = rc - rr
    logits = 0.1 * (policy_logratios - ref_logratios)
    expected = np.mean(np.log(1 + np.exp(-logits)))
    assert np.isclose(loss, expected, atol=1e-6)
_check("matches the DPO loss formula", _t1)

def _t2():
    pc = np.array([-1.0, -2.0]); pr = np.array([-1.5, -2.5])
    loss = dpo_loss(pc, pr, pc, pr, beta=0.1)  # policy == reference, no update yet
    assert np.isclose(loss, np.log(2), atol=1e-6)
_check("equals log(2) when the policy matches the reference (no update yet)", _t2)
''',
        },
        "torch": {
            "prompt": "Same, with torch tensors and gradients flowing through the policy log-probs.",
            "stub": '''def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # TODO
    raise NotImplementedError''',
            "solution": '''def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    return -torch.nn.functional.logsigmoid(logits).mean()''',
            "tests": '''
def _t1():
    pc = torch.tensor([-1.0, -0.5], requires_grad=True); pr = torch.tensor([-2.0, -1.5], requires_grad=True)
    rc = torch.tensor([-1.2, -0.6]); rr = torch.tensor([-1.8, -1.4])
    loss = dpo_loss(pc, pr, rc, rr, beta=0.1)
    policy_logratios = pc.detach() - pr.detach(); ref_logratios = rc - rr
    logits = 0.1 * (policy_logratios - ref_logratios)
    expected = torch.mean(torch.log(1 + torch.exp(-logits)))
    assert torch.isclose(loss, expected, atol=1e-5)
_check("matches the DPO loss formula", _t1)

def _t2():
    pc = torch.tensor([-1.0, -2.0], requires_grad=True); pr = torch.tensor([-1.5, -2.5], requires_grad=True)
    loss = dpo_loss(pc, pr, pc.detach(), pr.detach(), beta=0.1)
    assert torch.isclose(loss, torch.tensor(0.6931471805599453), atol=1e-5)
    loss.backward()
    assert pc.grad is not None and not torch.any(torch.isnan(pc.grad))
_check("equals log(2) when the policy matches the reference, and gradients flow", _t2)
''',
        },
    },
]

CONCEPTS_BY_ID = {c["id"]: c for c in CONCEPTS}
