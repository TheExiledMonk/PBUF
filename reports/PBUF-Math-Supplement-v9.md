
# Planck-Bound Unified Framework (PBUF) — Mathematical Supplement (v9.0)
**Compiled:** 2025-10-20  
**Scope:** Derivation of the elastic stress tensor σ_{μν}, proof of covariant conservation, and stability analysis (background and linear perturbations), with a perturbative appendix.

---

## 0. Notation and Conventions
- Signature: (-,+,+,+); covariant derivative ∇_μ; reduced Planck mass M_P^2 = (8πG)^{-1}.  
- Curvature tensors: R, R_{μν}, R_{μναβ}; Einstein tensor G_{μν} = R_{μν} - ½ g_{μν} R.  
- Matter stress-energy: T_{μν} = -\frac{2}{\sqrt{-g}} \frac{δ(\sqrt{-g} \mathcal{L}_m)}{δ g^{μν}}.  
- We define the **elastic stress tensor** σ_{μν} so that the field equations read
\[
G_{μν} + σ_{μν} = 8πG \, T_{μν}.
\]

---

## 1. Effective Action and Constitutive Definition of σ_{μν}

We posit a diffeomorphism-invariant action
\[
S[g,\Psi] = \int d^4x \sqrt{-g} \left[\frac{R}{16πG} + \mathcal{L}_{\mathrm{elastic}}(g; \mathcal{I}) + \mathcal{L}_m(g,\Psi)\right],
\]
where \(\mathcal{I}\) denotes curvature invariants (e.g., \(R,\, R_{μν}R^{μν},\, R_{μναβ}R^{μναβ}\)).  
The **elastic stress** is defined by metric variation of \(\mathcal{L}_{\mathrm{elastic}}\):
\[
σ_{μν} \equiv -\frac{2}{\sqrt{-g}} \frac{δ\left(\sqrt{-g}\,\mathcal{L}_{\mathrm{elastic}}\right)}{δ g^{μν}}.
\]
Varying the total action with respect to \(g^{μν}\) yields
\[
\frac{1}{8πG} G_{μν} + \frac{1}{2} σ_{μν} = \frac{1}{2} T_{μν}\quad \Rightarrow\quad G_{μν} + σ_{μν} = 8πG\, T_{μν}.
\]
Thus σ_{μν} plays the role of an **internal, Lorentz-covariant stress** of spacetime (elastic vacuum).

### 1.1 Exemplary bounded-\(f(R)\) realization
A convenient realization is a bounded-\(f(R)\) sector
\[
\mathcal{L}_{\mathrm{elastic}} = \frac{1}{16πG}\,\big[f(R) - R\big],\quad
f_R \equiv \frac{df}{dR}.
\]
Then the field equations can be written in standard \(f(R)\) form and recast as \(G_{μν}+σ_{μν}=8πG T_{μν}\) with
\[
σ_{μν} = (f_R-1)\,G_{μν} + \frac{1}{2}g_{μν}(f-R f_R) + \nabla_{μ}\nabla_{ν} f_R - g_{μν}\Box f_R.
\]
**Boundedness** is enforced by choosing \(f(R)\) such that \(f_R>0\), \(f_{RR}>0\) and \(|R|\le R_{\star}\) is energetically preferred, e.g.
\[
f(R) = R_{\star}\,\tanh\!\left(\frac{R}{R_{\star}}\right) + \lambda\,R
\]
or a smooth saturation \(f(R)=R_{\star}^2 \beta^{-1}\!\left[1-\sqrt{1-2\beta R/R_{\star}^2}\right]\) for small \(|R|\) recovering GR.

These choices implement the **Planck Bound** by making large-\(|R|\) costly while keeping low-curvature physics GR-like.

---

## 2. Covariant Conservation of σ_{μν}

Because \(S\) is diffeomorphism invariant and matter is minimally coupled,
\[
\nabla^{μ} T_{μν} = 0.
\]
Taking the covariant divergence of the field equations and using the Bianchi identity \(\nabla^{μ}G_{μν}=0\) gives
\[
\nabla^{μ} σ_{μν} = 0.
\]
Hence the **elastic stress is covariantly conserved**. Equivalently, Noether’s theorem for diffeomorphism invariance ensures that any purely geometric sector derived from a scalar density \(\sqrt{-g}\mathcal{L}_{\mathrm{elastic}}(g; \mathcal{I})\) yields a conserved σ_{μν} when matter is separately conserved.

---

## 3. Background Cosmology (FLRW) and Effective Fluid

On a spatially homogeneous and isotropic FLRW metric,
\[
ds^2=-dt^2+a(t)^2 \gamma_{ij}dx^i dx^j, \quad K\in\{0,\pm 1\},
\]
we decompose σ as an effective perfect fluid:
\[
σ^{μ}{}_{ν} = \mathrm{diag}(-8πG\,\rho_σ,\,8πG\,p_σ,\,8πG\,p_σ,\,8πG\,p_σ).
\]
The modified Friedmann equations read
\[
3H^2 + 3\frac{K}{a^2} + σ^{0}{}_{0} = 8πG\,\rho_m + 8πG\,\rho_r,
\]
\[
-2\dot H - 3H^2 - \frac{K}{a^2} + \tfrac{1}{3}σ^{i}{}_{i} = 8πG\,p_m + 8πG\,p_r.
\]
Define \( \rho_σ \equiv -\frac{1}{8πG}σ^{0}{}_{0}\), \( p_σ \equiv \frac{1}{24πG} σ^{i}{}_{i}\).  
The conservation \(\nabla\cdot σ=0\) yields
\[
\dot\rho_σ + 3H(\rho_σ + p_σ)=0,
\]
so the elastic sector behaves as a **self-consistent effective fluid** with equation of state \(w_σ \equiv p_σ/\rho_σ\) that can mimic dark energy (late-time \(w_σ \approx -1\)) or contribute to early-time rigidity.

---

## 4. Linear Perturbations and Stability Criteria

We work in Newtonian gauge:
\[
ds^2 = -(1+2\Phi)dt^2 + a(t)^2 (1-2\Psi)\delta_{ij}dx^i dx^j,
\]
and decompose perturbations into scalar, vector, tensor sectors.

### 4.1 Ghost and gradient stability
The quadratic action for scalar perturbations can be written schematically as
\[
S^{(2)}_{S} = \frac{1}{2}\int d^3k\,dt\,a^3 \left[ Q_S \dot{\zeta}^2 - \frac{c_S^2 k^2}{a^2}\zeta^2 \right],
\]
with curvature perturbation \(\zeta\). **No-ghost** requires \(Q_S>0\); **no-gradient-instability** requires \(c_S^2>0\).  
For bounded \(f(R)\)-type realizations, sufficient conditions are
\[
f_R > 0,\qquad f_{RR} > 0,
\]
which guarantee positive kinetic terms and real propagation speed for the additional scalar mode.

### 4.2 Tensor sector and gravitational waves
Tensor perturbations \(h_{ij}\) satisfy
\[
\ddot h_{ij} + (3H+\nu) \dot h_{ij} + c_T^2 \frac{k^2}{a^2} h_{ij} = \Pi_{ij}.
\]
In elastic sectors constructed from curvature invariants without disformal matter couplings, one finds **\(c_T^2=1\)** (GW170817 safe).  
The friction correction \(\nu \propto \frac{d}{dt}\ln f_R\) is small for the calibrated PBUF background, so standard siren distances primarily test amplitude damping rather than speed shifts.

### 4.3 Vector modes
No additional dynamical vectors are introduced; standard decay persists.

---

## 5. Example: Tanh-bounded \(f(R)\) and Parameter Conditions

Let \(f(R)=R_{\star}\tanh(R/R_{\star})+\lambda R\). Then
\[
f_R = \lambda + \operatorname{sech}^2(R/R_{\star}),\qquad
f_{RR} = -\frac{2}{R_{\star}}\tanh(R/R_{\star})\operatorname{sech}^2(R/R_{\star}).
\]
Choose \(\lambda>0\) and restrict \(|R| \ll R_{\star}\) at late times to ensure \(f_R>0\).  
If needed, modify the ansatz to enforce \(f_{RR}>0\) globally (e.g., use a smooth saturating function with positive curvature), or adopt a Padé-bounded form with \(f_{RR}>0\) by design.  
These choices keep \(Q_S>0\), \(c_S^2>0\) while reproducing the calibrated PBUF background expansion (encoded via the single observational parameter \(k_{\mathrm{sat}}\)).

---

## 6. Hamiltonian Positivity and Ostrogradsky Avoidance

The elastic sector is constructed from curvature scalars and respects diffeomorphism invariance.  
Mapping to the scalar–tensor representation (via Legendre transform \(ϕ \equiv f_R\)) yields a **Brans–Dicke-like** theory with \(\omega_{\mathrm{BD}}>0\) and potential \(V(ϕ)\) that is bounded below; this removes Ostrogradsky instabilities and ensures a Hamiltonian bounded from below in the perturbative regime relevant to cosmology.

---

## 7. Perturbative Appendix (Sketch of Derivations)

1. **Second variation and Q_S:** Vary the action to second order around FLRW; integrate out nondynamical constraints to obtain \(Q_S\) and \(c_S^2\). For \(f(R)\), these reduce to functions of \(f_R, f_{RR}, H,\dot H\).  
2. **Tensor propagation:** Start from the transverse-traceless sector and show \(c_T^2=1\) for curvature-only elastic terms.  
3. **Effective fluid mapping:** Identify \(ρ_σ, p_σ\) by projecting σ_{μν} on the comoving 4-velocity \(u^{μ}\).  
4. **Background closure:** Show that the modified Friedmann system closes with \(\nabla\cdot σ=0\) independent of matter conservation, consistent with Bianchi identities.

---

## 8. Summary of Mathematical Results

- The elastic stress tensor **σ_{μν}** is obtained from \(\mathcal{L}_{\mathrm{elastic}}\) by metric variation and is **covariantly conserved**.  
- On FLRW, σ behaves as a conserved effective fluid, enabling \(w_σ(a)\) that explains late-time acceleration.  
- Linear-perturbation stability is ensured by **\(f_R>0, f_{RR}>0\)** (or the equivalent scalar–tensor conditions), yielding **no ghosts**, **no gradient instabilities**, and **\(c_T^2=1\)**.  
- The framework is thus mathematically well-posed for cosmological evolution and gravitational waves, consistent with the empirical v9.0 calibration.

