Equation & Parameter Reference — PBUF Cosmology Framework

Repository: TheExiledMonk/PBUF
Version: 2025-10-22
Document purpose: Summarize all mathematical relations implemented in code, their physical meanings, and file origins.
Physics sources:

PBUF Unified Manuscript v9 (Sec. 2–3)

Proof Dossier v2

Planck 2018 Distance Priors

Eisenstein & Hu (1998)

Hu & Sugiyama (1996)

1. Cosmological Background — General Relativity (ΛCDM)
(1.1) Friedmann Equation
𝐸
2
(
𝑧
)
=
𝐻
2
(
𝑧
)
𝐻
0
2
=
Ω
𝑚
(
1
+
𝑧
)
3
+
Ω
𝑟
(
1
+
𝑧
)
4
+
Ω
𝑘
(
1
+
𝑧
)
2
+
Ω
Λ
E
2
(z)=
H
0
2
	​

H
2
(z)
	​

=Ω
m
	​

(1+z)
3
+Ω
r
	​

(1+z)
4
+Ω
k
	​

(1+z)
2
+Ω
Λ
	​


Meaning:
Dimensionless Hubble expansion factor normalized to 
𝐻
0
H
0
	​

.
Describes expansion of the Universe under GR with matter, radiation, curvature, and vacuum components.

Units: dimensionless
Implemented in: core/gr_models.py::E2()
Parameters:

𝐻
0
H
0
	​

: Hubble constant (km s⁻¹ Mpc⁻¹)

Ω
𝑚
Ω
m
	​

: present-day matter density fraction

Ω
𝑟
Ω
r
	​

: radiation fraction

Ω
𝑘
Ω
k
	​

: curvature parameter (0 in flat models)

Ω
Λ
=
1
−
Ω
𝑚
−
Ω
𝑟
−
Ω
𝑘
Ω
Λ
	​

=1−Ω
m
	​

−Ω
r
	​

−Ω
k
	​


(1.2) Hubble Parameter
𝐻
(
𝑧
)
=
𝐻
0
𝐸
2
(
𝑧
)
H(z)=H
0
	​

E
2
(z)
	​


Units: km s⁻¹ Mpc⁻¹
Meaning: Instantaneous expansion rate at redshift z.
Implemented in: core/gr_models.py::H()

(1.3) Comoving Distance
𝐷
𝑐
(
𝑧
)
=
𝑐
∫
0
𝑧
𝑑
𝑧
′
𝐻
(
𝑧
′
)
D
c
	​

(z)=c∫
0
z
	​

H(z
′
)
dz
′
	​


Meaning: Line-of-sight comoving separation between observer and redshift z.
Units: Mpc
Implemented in: gr_models.Dc()

(1.4) Luminosity Distance
𝐷
𝐿
(
𝑧
)
=
(
1
+
𝑧
)
 
𝐷
𝑐
(
𝑧
)
D
L
	​

(z)=(1+z)D
c
	​

(z)

Meaning: Effective distance inferred from luminosity measurements (e.g., SNe).
Units: Mpc
Implemented in: gr_models.DL()

(1.5) Distance Modulus
𝜇
=
5
log
⁡
10
 ⁣
(
𝐷
𝐿
10
 
p
c
)
μ=5log
10
	​

(
10 pc
D
L
	​

	​

)

Meaning: Apparent brightness measure for supernova fits.
Implemented in: gr_models.mu()

2. PBUF Extension — Elastic Spacetime Model
(2.1) Modified Friedmann Equation
𝐸
PBUF
2
(
𝑧
)
=
𝐸
Λ
CDM
2
(
𝑧
)
+
𝜎
eff
(
𝑧
;
𝑘
sat
,
𝛼
,
𝑅
max
⁡
,
𝜀
0
,
𝑛
𝜀
)
E
PBUF
2
	​

(z)=E
ΛCDM
2
	​

(z)+σ
eff
	​

(z;k
sat
	​

,α,R
max
	​

,ε
0
	​

,n
ε
	​

)

Meaning:
Adds an elastic energy density term that saturates near the Planck curvature bound.
Represents finite spacetime rigidity preventing singularities and replacing dark energy with elastic self-restoring behavior.

Units: dimensionless
Implemented in: core/pbuf_models.py::E2()

Current implementation note:
The analytic manuscript form (Ωσ(a) = α [1 − e^(−a/Rₘₐₓ)]) is not yet numerically realized.
Code currently uses a phenomenological placeholder via pbuf_sigma.sigma_eff() and pbuf_evolution.evolve_params().

(2.2) Saturation Factor
𝑘
sat
=
dimensionless elasticity parameter
k
sat
	​

=dimensionless elasticity parameter

Meaning:
Controls how quickly the vacuum strain saturates at the Planck limit.
Acts as the only new free parameter beyond ΛCDM in the low-energy regime.
When 
𝑘
sat
=
1
k
sat
	​

=1, PBUF reduces to ΛCDM.

Implemented in: pbuf_models._saturation_factor()

(2.3) Elastic Energy Component (from manuscript)
Ω
𝜎
(
𝑎
)
=
𝛼
(
1
−
𝑒
−
𝑎
/
𝑅
max
⁡
)
Ω
σ
	​

(a)=α(1−e
−a/R
max
	​

)

Meaning:
Phenomenological energy density capturing the geometric elasticity of spacetime.
α sets amplitude, 
𝑅
max
⁡
R
max
	​

 sets saturation scale.
This term replaces Λ in the modified Friedmann equation.

Parameters:

α — elasticity amplitude (dimensionless)

Rₘₐₓ — maximum curvature scale in comoving units

ε₀, n_ε — optional evolution exponents for adaptive strain decay

(2.4) Radiation Correction (implemented)
Ω
𝑟
(
1
+
𝑧
)
4
(
𝑘
sat
−
1
)
Ω
r
	​

(1+z)
4
(k
sat
	​

−1)

Meaning:
Small correction ensuring early-time PBUF expansion recovers standard radiation behavior when 
𝑘
sat
→
1
k
sat
	​

→1.
Implemented in: pbuf_models.E2()

3. CMB Distance Priors
(3.1) Recombination Redshift 
𝑧
∗
z
∗
	​


Options:

HS96 (Hu & Sugiyama, 1996):

𝑧
∗
=
1048
(
1
+
0.00124
 
Ω
𝑏
−
0.738
)
(
1
+
𝑔
1
Ω
𝑚
𝑔
2
)
z
∗
	​

=1048(1+0.00124Ω
b
−0.738
	​

)(1+g
1
	​

Ω
m
g
2
	​

	​

)

with

𝑔
1
=
0.0783
 
Ω
𝑏
−
0.238
/
(
1
+
39.5
 
Ω
𝑏
0.763
)
g
1
	​

=0.0783Ω
b
−0.238
	​

/(1+39.5Ω
b
0.763
	​

),

𝑔
2
=
0.560
/
(
1
+
21.1
 
Ω
𝑏
1.81
)
g
2
	​

=0.560/(1+21.1Ω
b
1.81
	​

)

EH98 proxy: 
𝑧
∗
=
1.04
 
𝑧
𝑑
z
∗
	​

=1.04z
d
	​


PLANCK18 fit (default):

𝑧
∗
=
1089.80
(
Ω
𝑏
ℎ
2
0.02237
)
−
0.004
(
Ω
𝑚
ℎ
2
0.1424
)
0.010
z
∗
	​

=1089.80(
0.02237
Ω
b
	​

h
2
	​

)
−0.004
(
0.1424
Ω
m
	​

h
2
	​

)
0.010

Meaning:
Approximation for photon decoupling redshift at last scattering.
Implemented in: cmb_priors.z_recombination()

(3.2) Drag Epoch 
𝑧
𝑑
z
d
	​

 (Eisenstein & Hu 1998)
𝑧
𝑑
=
1291
 
(
Ω
𝑚
ℎ
2
)
0.251
[
1
+
0.659
 
(
Ω
𝑚
ℎ
2
)
0.828
]
−
1
[
1
+
𝑏
1
(
Ω
𝑏
ℎ
2
)
𝑏
2
]
z
d
	​

=1291(Ω
m
	​

h
2
)
0.251
[1+0.659(Ω
m
	​

h
2
)
0.828
]
−1
[1+b
1
	​

(Ω
b
	​

h
2
)
b
2
	​

]

where

𝑏
1
=
0.313
(
Ω
𝑚
ℎ
2
)
−
0.419
[
1
+
0.607
(
Ω
𝑚
ℎ
2
)
0.674
]
b
1
	​

=0.313(Ω
m
	​

h
2
)
−0.419
[1+0.607(Ω
m
	​

h
2
)
0.674
]

𝑏
2
=
0.238
(
Ω
𝑚
ℎ
2
)
0.223
b
2
	​

=0.238(Ω
m
	​

h
2
)
0.223

Meaning:
Epoch when baryons were released from photon drag — used in BAO sound horizon 
𝑟
𝑠
(
𝑧
𝑑
)
r
s
	​

(z
d
	​

).
Implemented in: cmb_priors.z_drag()

(3.3) Sound Speed of Photon–Baryon Fluid
𝑐
𝑠
(
𝑎
)
=
𝑐
3
(
1
+
𝑅
𝑏
(
𝑎
)
)
,
𝑅
𝑏
(
𝑎
)
=
3
 
Ω
𝑏
4
 
Ω
𝛾
 
𝑎
c
s
	​

(a)=
3(1+R
b
	​

(a))
	​

c
	​

,R
b
	​

(a)=
4Ω
γ
	​

3Ω
b
	​

	​

a

Meaning:
Instantaneous propagation speed of acoustic waves in early plasma.
Implemented in: cmb_priors.sound_horizon_a()

(3.4) Comoving Sound Horizon
𝑟
𝑠
(
𝑧
)
=
∫
0
𝑎
(
𝑧
)
𝑐
𝑠
(
𝑎
)
𝑎
2
𝐻
(
𝑎
)
 
𝑑
𝑎
r
s
	​

(z)=∫
0
a(z)
	​

a
2
H(a)
c
s
	​

(a)
	​

da

Meaning:
Comoving distance traveled by sound waves before decoupling.
Units: Mpc
Implemented in: cmb_priors.sound_horizon_a()

(3.5) Angular Diameter Distance
𝐷
𝐴
(
𝑧
)
=
𝐷
𝑀
(
𝑧
)
1
+
𝑧
,
𝐷
𝑀
(
𝑧
)
=
𝑐
∫
0
𝑧
𝑑
𝑧
′
𝐻
(
𝑧
′
)
D
A
	​

(z)=
1+z
D
M
	​

(z)
	​

,D
M
	​

(z)=c∫
0
z
	​

H(z
′
)
dz
′
	​


Implemented in: cmb_priors.D_A() and _comoving_transverse_distance()

(3.6) Acoustic Angular Scale
𝜃
∗
=
𝑟
𝑠
(
𝑧
∗
)
𝐷
𝑀
(
𝑧
∗
)
θ
∗
	​

=
D
M
	​

(z
∗
	​

)
r
s
	​

(z
∗
	​

)
	​

100
 
𝜃
∗
=
100
×
𝜃
∗
100θ
∗
	​

=100×θ
∗
	​


Meaning:
Angular size of the sound horizon at recombination.
Implemented in: cmb_priors.theta_star()

(3.7) Acoustic Scale
ℓ
𝐴
=
𝜋
𝐷
𝑀
(
𝑧
∗
)
𝑟
𝑠
(
𝑧
∗
)
ℓ
A
	​

=π
r
s
	​

(z
∗
	​

)
D
M
	​

(z
∗
	​

)
	​


Meaning:
Multipole number corresponding to the CMB acoustic peak.
Implemented in: cmb_priors.l_A()

(3.8) Shift Parameter
𝑅
=
Ω
𝑚
𝐻
0
𝑐
(
1
+
𝑧
∗
)
𝐷
𝐴
(
𝑧
∗
)
R=
Ω
m
	​

	​

c
H
0
	​

	​

(1+z
∗
	​

)D
A
	​

(z
∗
	​

)

Meaning:
Dimensionless measure encapsulating the geometry of the Universe for CMB fits.
Implemented in: cmb_priors.shift_parameter()

4. BAO Observables
(4.1) Volume-Averaged Distance
𝐷
𝑉
(
𝑧
)
=
[
𝐷
𝑀
2
(
𝑧
)
𝑐
𝑧
𝐻
(
𝑧
)
]
1
/
3
D
V
	​

(z)=[D
M
2
	​

(z)
H(z)
cz
	​

]
1/3

Meaning:
Spherically averaged distance scale used in isotropic BAO analyses.
Implemented in: bao_background._D_V()

(4.2) Isotropic BAO Ratios
𝐷
𝑉
(
𝑧
)
𝑟
𝑠
(
𝑧
𝑑
)
r
s
	​

(z
d
	​

)
D
V
	​

(z)
	​


Implemented in: bao_background.bao_distance_ratios()

(4.3) Anisotropic BAO Ratios
𝐷
𝑀
(
𝑧
)
𝑟
𝑠
(
𝑧
𝑑
)
,
𝐻
(
𝑧
)
 
𝑟
𝑠
(
𝑧
𝑑
)
r
s
	​

(z
d
	​

)
D
M
	​

(z)
	​

,H(z)r
s
	​

(z
d
	​

)

Meaning:
Transverse and radial BAO scaling ratios.
Implemented in: bao_background.bao_anisotropic_ratios()

5. Derived Statistical Quantities
(5.1) χ² (Chi-Squared)
𝜒
2
=
(
𝑥
model
−
𝑥
data
)
𝑇
𝐶
−
1
(
𝑥
model
−
𝑥
data
)
χ
2
=(x
model
	​

−x
data
	​

)
T
C
−1
(x
model
	​

−x
data
	​

)

Meaning:
Goodness-of-fit metric comparing model predictions to observed priors or datasets.
Implemented in: cmb_priors.chi2_cmb(), fit_core/statistics.py::chi2_generic()

(5.2) Information Criteria
𝐴
𝐼
𝐶
=
𝜒
2
+
2
𝑘
,
𝐵
𝐼
𝐶
=
𝜒
2
+
𝑘
ln
⁡
𝑁
AIC=χ
2
+2k,BIC=χ
2
+klnN

Meaning:
Model comparison metrics penalizing complexity.
k = number of free parameters, N = data points.
Implemented in: fit_joint._metrics_block() and will move to fit_core/statistics.py.

(5.3) Degrees of Freedom
dof
=
𝑁
points
−
𝑁
params
dof=N
points
	​

−N
params
	​


Used for reporting χ²/dof and p-values.

6. Derived Constants & Conversions
Symbol	Meaning	Implementation
c	Speed of light = 299 792.458 km/s	config/constants.py::C_LIGHT

𝑇
CMB
T
CMB
	​

	2.7255 K	config/constants.py::TCMB

𝑁
eff
N
eff
	​

	Effective relativistic species = 3.046	config/constants.py::NEFF

𝜔
𝛾
ℎ
2
=
2.469
×
10
−
5
(
𝑇
/
2.7255
)
4
ω
γ
	​

h
2
=2.469×10
−5
(T/2.7255)
4
	Photon density today	constants.omega_gamma_h2()
7. Parameter Summary Table
Parameter	Meaning	Type / Units	Appears In
H₀	Hubble constant today	km s⁻¹ Mpc⁻¹	All background models
Ωₘ₀	Matter density fraction	dimensionless	GR, PBUF
Ωᵣ₀	Radiation density fraction	dimensionless	GR, PBUF
Ωₖ₀	Curvature fraction	dimensionless	GR
Ω_Λ₀	Dark energy fraction (flat fix)	dimensionless	GR
Ω_b h²	Physical baryon density	dimensionless	CMB, BAO
n_s	Scalar spectral index	dimensionless	CMB priors
α	PBUF elasticity amplitude	dimensionless	PBUF
Rₘₐₓ	Saturation length scale	dimensionless	PBUF
ε₀	Elasticity bias term	dimensionless	PBUF
n_ε	Elasticity evolution exponent	dimensionless	PBUF
k_sat	Elastic saturation coefficient	dimensionless	PBUF
z*	Recombination redshift	dimensionless	CMB
z_d	Baryon drag epoch	dimensionless	BAO
r_s(z)	Sound horizon	Mpc	CMB, BAO
D_A, D_M, D_L, D_V	Distance measures	Mpc	CMB, BAO, SN
θ*, ℓ_A, R	CMB distance priors	dimensionless	CMB
μ	Distance modulus	mag	SN
8. Equation Provenance Mapping
Equation	Source	Code File
Friedmann / H(z)	General Relativity	core/gr_models.py
Elastic correction	PBUF Manuscript Eq. (3.4)	core/pbuf_models.py
z* fit (PLANCK18)	Planck 2018 App. C	core/cmb_priors.py
z_d (EH98)	Eisenstein & Hu 1998 Eq. (4–5)	core/cmb_priors.py
r_s integral	Standard Cosmology	core/cmb_priors.py
D_V, D_M ratios	BAO (Anderson et al. 2014)	core/bao_background.py
χ²	Statistical Definition	fit_core/statistics.py
AIC/BIC	Akaike, Schwarz	fit_core/statistics.py
9. Integration Validation Rules

When the dev-agent documents or reimplements modules, it must:

Cross-reference each implemented function with its corresponding equation in this table.

Ensure all constants and units are documented inline.

For every function returning physical quantities (r_s, D_M, etc.), list:

Equation tag (e.g., Eq. 3.4),

Physical meaning,

Default units,

Dependencies (parameters, constants).

Link back to documents/PBUF-Math-Supplement-v9.pdf for full derivations.

✅ Summary

This equation reference unifies the theoretical and computational layers:

ΛCDM equations are textbook GR.

PBUF adds a physically motivated saturation term with geometric elasticity.

CMB & BAO follow Planck 2018 + EH98 standards.

Statistical layer (χ², AIC/BIC) provides comparable, consistent fit metrics.