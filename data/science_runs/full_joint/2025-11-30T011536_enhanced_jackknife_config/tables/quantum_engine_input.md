## Quantum Engine Input Configuration

Complete configuration of quantum engine parameters and thermal physics settings for PBUF models, ensuring full reproducibility of quantum cosmology calculations.

| Parameter | Value | Unit | Description | Source |
| --- | --- | --- | --- | --- |
| === PBUF === |  |  |  |  |
| Regulator Type | — |  | Type of UV regularization scheme | quantum_engine |
| Field Content | — |  | Number and type of quantum fields | quantum_engine |
|  |  |  |  |  |
| UV Cutoff (f_cut) | 0.00e+00 | GeV | Maximum frequency scale for quantum modes | quantum_engine |
| Coupling Scale (f_coup) | 0.00e+00 |  | Quantum-gravity coupling strength | quantum_engine |
|  |  |  |  |  |
| ε₀ Source | — |  | Source of initial quantum vacuum energy | quantum_engine |
|  |  |  |  |  |
| α Value | 0.000 |  | Quantum deformation parameter | quantum_engine |
|  |  |  |  |  |
| Maximum Rigidity (R_max) | 0.00e+00 | GeV⁻¹ | Maximum spacetime rigidity scale | quantum_engine |
|  |  |  |  |  |
| Ω Normalization | — |  | Cosmological parameter normalization scheme | quantum_engine |
|  |  |  |  |  |
| σ Rescale | 0.000 |  | Rescaling factor for matter fluctuations | quantum_engine |
|  |  |  |  |  |
| LUT Type | Bootstrap |  | Type of lookup table used | bootstrap |
| LUT Version | — |  | Version identifier for thermal table | bootstrap |
| Interpolation Method | — |  | Interpolation scheme for thermal quantities | thermal_table |
|  |  |  |  |  |
| T_min | — | GeV | Minimum temperature in thermal table | thermal_table |
| T_max | — | GeV | Maximum temperature in thermal table | thermal_table |
| N_T Points | — |  | Number of temperature grid points | thermal_table |
|  |  |  |  |  |
| Normalization Mode | — |  | Parameter normalization approach | quantum_engine |

**Footnotes:**
1. Parameter: Quantum engine configuration parameter name
2. Value: Numerical value or setting used in the calculation
3. Unit: Physical units (if applicable)
4. Description: Brief explanation of the parameter's role in the quantum engine
5. Source: Origin of the parameter value (quantum_engine, bootstrap, thermal_table, etc.)
6. Regulator Type: UV regularization scheme (exponential, hard cutoff, etc.)
7. Field Content: Number and types of quantum fields included
8. f_cut: UV cutoff frequency scale for quantum mode integration
9. f_coup: Quantum-gravity coupling strength parameter
10. ε₀ Source: Source of initial quantum vacuum energy density
11. α Value: Quantum deformation parameter controlling departure from GR
12. R_max: Maximum spacetime rigidity scale (inverse energy)
13. LUT Type: Type of lookup table used for thermal quantities
14. Bootstrap: Self-consistent thermal field calculation method
