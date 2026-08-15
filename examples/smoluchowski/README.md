# Multicomponent coagulation: two and a half days of direct summation in four seconds

[`examples/smoluchowski_coagulation.py`](../smoluchowski_coagulation.py) reproduces the reference experiment of Matveev–Zheltkov–Tyrtyshnikov–Smirnov ([JCP 2016](https://doi.org/10.1016/j.jcp.2016.04.025)) — the two-component Smoluchowski coagulation equation on a $1000^2$ grid — reaching the paper's error 2.2e-3 at TT rank 13 in **4.5 s**, where the paper reports 1024 s for the same TT scheme and **215 580 s (2.5 days)** for the direct implementation of that scheme.

![The run as its own log: ranks, density and error against the analytic solution, next to the coagulating cloud](../../docs/media/smoluchowski_run.gif)

## The problem

A population of particles carrying $d$ conserved components — the concentration $n(\bar v, t)$, $\bar v = (v_1,\dots,v_d)$ — coagulates pairwise at a rate $K$:

$$
\frac{\partial n(\bar v,t)}{\partial t}
= \frac12 \int_0^{v_1}\!\!\cdots\!\int_0^{v_d} K(\bar v - \bar u;\, \bar u)\, n(\bar v - \bar u)\, n(\bar u)\, d\bar u
\;-\; n(\bar v) \int_0^{\infty}\!\!\cdots\!\int_0^{\infty} K(\bar u;\, \bar v)\, n(\bar u)\, d\bar u .
$$

The gain term counts the particles of size $\bar v$ assembled from $\bar u$ and $\bar v - \bar u$; the loss term counts the ones of size $\bar v$ eaten by anything. On a grid of $N$ nodes per component the gain term is a $2d$-fold sum: $O(N^{2d})$ per time step, $10^{12}$ operations at the modest $d=2$, $N=1000$. That is the 215 580 s.

## Two facts, composed

**The gain term is a lower-triangular convolution.** On a uniform grid it is a plain discrete convolution of zero-padded arrays, so the FFT does it in $O(N\log N)$ per axis. The trapezoidal quadrature weights are not in the way: halving the $v_k = 0$ slice of *both* factors turns the plain convolution into exactly the trapezoidal one, because the $\tfrac12$ endpoint weights of $\int_0^{v_k}$ land on $j = 0$ and $j = i$ by themselves. Written out for one axis,

$$
\tilde f_0 = \tfrac12 f_0,\quad \tilde g_0 = \tfrac12 g_0
\;\Longrightarrow\;
(\tilde f * \tilde g)_i
= \tfrac12 f_i g_0 + \tfrac12 f_0 g_i + \sum_{j=1}^{i-1} f_{i-j} g_j
= \sum_{j=0}^{i} w_j^{(i)} f_{i-j} g_j ,
$$

which is the trapezoidal rule on $[0, v_i]$, exactly.

**Every step of that is a per-mode operation.** The FFT along mode $k$, the elementwise product, the truncation back to the first $N$ entries, the zeroing of the $v_k = 0$ slice — each one touches a single TT core and leaves the rest of the train alone. What is left is the elementwise product of two TT tensors, whose cores are the Kronecker products of the factors' cores over the rank indices. So the whole convolution never leaves the format:

```python
for k in range(d):
    a = f_cores[k].astype(complex);  a[:, 0, :] *= 0.5      # trapezoid
    b = g_cores[k].astype(complex);  b[:, 0, :] *= 0.5
    a = np.fft.fft(a, n=2 * nk, axis=1)                     # zero-pad + FFT
    b = np.fft.fft(b, n=2 * nk, axis=1)
    q = np.einsum("aib,cid->acibd", a, b).reshape(...)      # ranks multiply
    q = np.fft.ifft(q, axis=1) * h
    q = q[:, :nk, :];  q[:, 0, :] = 0.0                     # drop wrap-around
```

Cost: $O(d R^4 N \log N)$ — $2R^2$ forward transforms, $R^4$ inverse ones, one rounding of a rank-$R^2$ train. That is [`tt.algs.smoluchowski.trapezoidal_convolution`](../../tt/algs/smoluchowski.py), Algorithm 1 of the paper.

**The kernel has to be separable**, and enters as a list of rank-1 terms $K(\bar u;\bar v) = \sum_\alpha k^v_\alpha(\bar v) k^u_\alpha(\bar u)$. The gain then splits into one convolution per term, and — the part that is easy to miss — the loss term collapses from a $d$-fold integral *at every grid point* into one **scalar** quadrature per term:

$$
L_2(\bar v) = \sum_\alpha k^v_\alpha(\bar v) \int k^u_\alpha(\bar u)\, n(\bar u)\, d\bar u .
$$

Time stepping is the paper's eq. (5), the explicit midpoint predictor–corrector, second order. The log panel of the animation prints the TT rank after *both* stages, which is where the method's real cost lives: the predictor inflates the ranks (every Hadamard product multiplies them) and the rounding pulls them back.

## Oracle

For $K \equiv 1$ and $n_0 = ab\,e^{-a v_1 - b v_2}$ the paper's eq. (18) is exact:

$$
n(v_1,v_2,t) = \frac{ab\,e^{-a v_1 - b v_2}}{(1+t/2)^2}\;
I_0\!\left(2\sqrt{\frac{ab\,v_1 v_2\,t}{t+2}}\right),
\qquad
N(t) = \int\!\!\int n = \frac{1}{1+t/2}\ \text{exactly.}
$$

Both are checked every step, and the totals are what the log's `density`/`exact` columns compare.

## What it reaches

Paper setup: $d=2$, $K\equiv 1$, $V_{\max} = 100$, $T = 10$, $a=b=1$, $\varepsilon = 10^{-6}$. Measured here (numpy backend, one core of an M-series laptop) against Table 1 of the paper:

| $N$ | $\tau$ | error, here | error, paper | rank, here | rank, paper | time, here | TT, paper | direct, paper |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.1 | 6.3e-2 | 1.4e-1 | 11 | — | 0.4 s | — | ~22 s¹ |
| 500 | 0.1 | 4.1e-3 | — | 13 | — | 2.1 s | — | 12 225 s |
| 1000 | 0.1 | **2.24e-3** | 2.2e-3 | **13** | 13 | **4.5 s** | 1 024 s | **215 580 s** |
| 2000 | 0.05 | 6.03e-4 | 5.0e-4 | 12 | — | 12.7 s | 2 492 s | — |

¹ extrapolated from the paper's own $O(N^4)$ direct timings, not measured.

The error column is the relative Frobenius distance to eq. (18) on the whole grid; the total density is an order more accurate again (1.8e-4 at $N = 1000$), because its leading quadrature errors cancel. The scheme's advertised $O(h^2 + \tau^2)$ is visible directly: halving $h$ and $\tau$ together on $V_{\max}=20$, $T=1$ divides the error by 3.69 and then 3.86.

Two remarks on the comparison. The paper's timings are its own machine and its own decade, so the 230× against its TT run is not a claim about the algorithm — the algorithm is the same one. The 48 000× against the direct implementation *is*, and it is the number that matters: it is the same scheme, the same grid, the same answer, and it is the difference between a coffee break and a long weekend.

## Additive kernel, and what is missing

`--kernel additive` runs $K = \sum_i u_i + \sum_i v_i$ (rank 2). There is no analytic solution here, but there are two exact identities on the unbounded domain — the mass $\int (v_1+v_2)\,n$ is conserved and $N(t) = N_0 e^{-M_0 t}$ — and the script checks both. On the truncated box mass does leave through the top, which is the model, not a bug: at $V_{\max} = 40$, $T = 0.2$ the density falls by 33% while the mass drifts by 3.2e-3.

The **ballistic kernel** of the paper's eq. (17),

$$
K = \left(\Big(\textstyle\sum_i u_i\Big)^{1/3} + \Big(\textstyle\sum_i v_i\Big)^{1/3}\right)^{2}
\sqrt{\frac{1}{\sum_i u_i} + \frac{1}{\sum_i v_i}},
$$

is **not implemented**. It is not separable, so using it needs a separable approximation of the $2d$-dimensional $K$ built by cross approximation and then split into $(k^v, k^u)$ pairs — plus an honest account of the error that approximation adds. That work is not done here, and is recorded as missing rather than papered over.

Other limits worth stating plainly: the scheme is second order and no better, so three digits in the profile need a fine grid rather than a tighter `eps`; the ranks are held down by rounding and by nothing else (no theorem promises they stay small — the printed rank is the only monitor); the grid must be uniform, because the FFT convolution *is* the uniform grid; and nothing enforces $n \ge 0$.

## Running it

```
python examples/smoluchowski_coagulation.py                          # the table row above
python examples/smoluchowski_coagulation.py --N 2000 --tau 0.05      # the next one
python examples/smoluchowski_coagulation.py --kernel additive --T 0.3 --tau 0.01
python examples/smoluchowski_coagulation.py --gif docs/media/smoluchowski_run.gif
```

The acceptance versions of both oracles live in `tests/test_examples.py`; the algorithm itself is pinned in `tests/test_smoluchowski.py`, including the convolution against a brute-force dense quadrature written from the definition (agreement 1.1e-15).
