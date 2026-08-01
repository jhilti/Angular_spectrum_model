# Physics audit and ADE design guidance

## Audit conclusion

The code/formula audit and numerical tests found the documented conventions
internally consistent for a **linear, small-signal, fixed-interface acoustic
propagation model**. The repository can model relative focal position, spatial
pressure shape, interface timing, and elastic-PP transmission. It supports a
calibrated linear pulse-echo chain only after independent bench validation.

It is **not** a first-principles acoustic droplet-ejection (ADE) solver. It does
not yet solve free-surface deformation, jet formation, pinch-off, satellites,
streaming, cavitation, heating, or nonlinear propagation. Consequently, it
must not turn an uncalibrated ADC count or a pulser voltage setting directly
into a predicted drop volume or ejection threshold.

The core propagation convention was checked independently:

- phasors use `exp(-i omega t)` and forward waves use `exp(+i kz z)`;
- the selected square-root branch is outgoing or evanescent-decaying;
- the focused-aperture phase and NumPy inverse-FFT conjugation have the correct
  signs;
- the elastic plate satisfies normal displacement, normal traction, and zero
  shear traction at both fluid interfaces;
- P/SV conversion, reverse incidence, and internal plate reverberation are
  included;
- direct lossless plate solves conserve power and satisfy reciprocity to
  numerical precision.

The band-limited propagation follows the numerical principle described by
[Matsushima and Shimobaba](https://doi.org/10.1364/OE.17.019662).

## Corrected physics and numerical defects

| Correction | Why it matters |
|---|---|
| Temperature-dependent water density and sound speed | The old interface timing and water-path phase were wrong away from 22 °C; at 40 °C the 25.3 mm round trip was late by about 0.9 µs. The mixture interpolation is now continuous at zero DMSO. |
| Correct paraxial refraction ratios | The old estimate inverted sound-speed ratios. For the 80% DMSO example with a 20 mm water gap, it estimated 5.46 mm after PP instead of the correct 3.62 mm ray estimate. |
| Meniscus intensity retains aperture pressure | Doubling configured aperture pressure now produces four times the reported W/m² instead of no change. |
| Correct layered and two-way angular-spectrum support | Cumulative lateral walkoff is checked across the complete layered path, rather than checking water and liquid independently. Bundled pulse-echo grids were widened accordingly. |
| Order-specific liquid-cavity propagation | Echo orders use `2h`, `4h`, `6h`, … propagation and their own cumulative band limit. The prior geometric power incorrectly reused the `2h` mask for every later order. |
| Finite retained liquid-cavity echo train | Nominal surface-echo orders that fit inside the guarded record are summed; later orders are omitted instead of deliberately wrapping to the inverse-FFT start. Bundled pulse plots retain the robust first surface order. A gated record ending before it can still return the plate-front response. |
| Guarded pulse record and smooth DC rejection | The certificate-shaped response no longer has a nonphysical DC component. Unknown phase, hard frequency selection, oblique delay, and plate ring-down still require a convergence check. |
| Passive viscoelastic input check | Independently entered P- and S-wave attenuation can imply active bulk gain. Such combinations are now rejected. |
| Passive electrical impedance check | A negative-resistance source or probe can no longer silently create energy. |
| Explicit receive loading | A measured receiver input impedance can now be applied as the receive-side transducer/load divider. |
| Exact unique-radius plate mode | `radial_samples=None` now solves every unique FFT radius once without interpolation error. |
| Grazing-incidence fluid limit | The removable `0/0` singularity at a common grazing angle now has its finite analytical limit. |
| Honest meniscus fields | The CW sweep returns total fluid-side pressure, normal velocity, and first-order displacement. Its phase is identified as a pressure-amplitude optimum, not a strict intensity optimum. |
| Survey excitation metadata | Frequency, tone length, voltage setting, and sample-index start are retained, but explicitly not treated as calibrated terminal values. |

The Fast plate interpolation remains an approximation for speed. It is highly
accurate over the angular spectrum that carries the default focused beam, but
can violate pointwise passivity near very sharp, unused critical-angle or
guided-mode features. Use `plate_radial_samples=None` for a final convergence
check when resonant plate behavior is central.

The present power-law attenuation adds amplitude decay but no corresponding
causal dispersion. It is therefore a phenomenological narrowband model. For
broadband phase or group-delay work, replace it with measured complex
wavenumbers or a causal relaxation/viscoelastic model fitted over the band of
interest.

The certificate supplies magnitude, not system phase. Its zero-phase Gaussian
approximation and the active-frequency threshold are symmetric operations and
can create sub-percent to percent-level pre-ringing around strong echoes. The
liquid-cavity echo count uses guarded central-ray delays and prevents known
future surface orders from being intentionally periodized, but it cannot make
an unknown zero-phase system response causal. Oblique rays and the steady-state
elastic-plate response can also ring later. For timing work, vary record length,
spectral threshold, radial sampling, grid size, and measured loss; accept a
feature only if its time and amplitude converge.

## What the present geometry says

For 80 vol.% DMSO at 22 °C, 10 MHz, a 13 mm aperture, 25.4 mm nominal focus,
and a 4.22 mm liquid layer, the linear screening calculation gives:

| Quantity | Value | Interpretation |
|---|---:|---|
| DMSO-mixture sound speed | 1632.0 m/s | Literature interpolation, not a sample measurement |
| Wavelength | 163.2 µm | Acoustic length scale in the liquid |
| F-number | 1.954 | Moderately focused aperture |
| Homogeneous `1.02 F# lambda` intensity-FWHM proxy | 0.325 mm | Ignores layered refraction |
| Layered-model meniscus intensity FWHM | about 0.303 mm | Linear one-way result at the modeled optimum |
| Homogeneous-proxy sphere volume | 18.0 nL | Geometric scale only, not predicted drop volume |
| One-cycle duration | 0.100 µs | Ideal electrical duration; acoustic ring-down is unknown |
| Liquid cavity round trip | 5.172 µs / 51.7 cycles | The first cavity return cannot overlap a perfect one-cycle drive |
| Fixed-probe water-gap optimum | about 19.1 mm | Nominal linear 10 MHz `|p|²` optimum at the 4.22 mm meniscus |

The historical focused-beam experiments found that drop diameter follows focal
spot size only as an order-of-magnitude relationship, and that sufficient
surface forcing is still required for ejection
([Elrod et al.](https://doi.org/10.1063/1.342663),
[Hadimioglu et al.](https://doi.org/10.1109/ULTSYM.1992.275823)). At this
F-number, therefore, 0.325 mm and 18 nL are rough homogeneous-medium design
scales—not promises. The layered simulation's approximately 0.303 mm intensity
FWHM is the better linear spot estimate, but still not a droplet diameter.

The CW meniscus-interference fringes must not be used to choose the height for
a one-cycle ping. At 4.22 mm, the first liquid-cavity return arrives 51.7
cycles later. CW fringes become relevant only when the acoustic ring-down or
tone burst lasts long enough for coherent returns to overlap. The first-pulse
surface response and the later cavity train should be evaluated separately.
In the idealized 2–3 mm, lossless, planar, pressure-phase-optimized sweep, the
calculated CW interference contribution spans about −0.72 to +0.60 dB with a
dominant 0.0835 mm period. Meniscus curvature, loss, and a fixed single-element
focus can reduce or shift those fringes.

## Recommended real-device workflow

### 1. Establish geometry and material properties

1. Measure the transducer-to-PP front distance mechanically. Use absolute echo
   time only as a cross-check because it can contain trigger and cable delay.
2. Derive PP thickness and liquid height primarily from **differences** between
   interface arrival times.
3. Record temperature at every acquisition. Measure the actual mixture sound
   speed and density if concentration accuracy matters; DMSO/water properties
   are nonlinear in concentration and DMSO is hygroscopic.
4. Measure viscosity and surface tension at the actual concentration and
   temperature. Record wetting/contact angle, static meniscus curvature, and
   dissolved-gas/degassing protocol; these govern jetting, satellites, and
   cavitation but are absent from the linear solver.
5. Measure PP density, longitudinal speed, shear speed, and frequency-dependent
   attenuation. Extruded PP can be anisotropic and viscoelastic, whereas this
   solver currently assumes a homogeneous isotropic plate.
6. Repeat pulse echo at several known liquid heights. A valid model should fit
   all heights with one set of material and system parameters, rather than a
   separate fitted correction for each trace.

### 2. Define what “150 V” means

Measure voltage and current at the probe connector with the real cable and
load. Keep these quantities distinct:

- generator or pulser setting;
- open-circuit Thevenin voltage;
- loaded peak terminal voltage;
- peak-to-peak voltage;
- receiver input termination.

For illustration only, a 150 V **peak open-circuit** one-cycle sine from a
50 Ω source into a fictitious 50 Ω probe produces 75 V peak at the probe and
5.625 µJ net load energy. A 150 V **peak loaded** sine in the same fictitious
resistor would absorb 22.5 µJ. A real piezoelectric probe is reactive and
frequency dependent, so neither number should be used until its complex
impedance has been measured.

Before any high-voltage sweep, verify the probe, pulser, connector, cable, and
termination peak-voltage/current/duty ratings under the loaded impedance. Use
current limiting, a discharge/bleeder path, HV-rated differential probing,
shielding, an enclosure/interlock, discharged-state checks, low initial PRF,
and thermal stop limits. Follow the SDS for the actual DMSO mixture and verify
material compatibility; contain splashes and aerosols created by cavitation or
spray. These are device-safety requirements, not outputs of this model.

### 3. Calibrate the acoustic field at safe drive

1. Measure complex probe impedance versus frequency while mounted and loaded
   as in the ejector.
2. Map the low-amplitude incident field in a matched surrogate liquid stack
   with a calibrated hydrophone whose active diameter is well below the
   approximately 0.30 mm spot; correct for spatial averaging. A needle
   hydrophone at the actual free surface changes that boundary, so validate
   the real meniscus separately with high-speed or laser-optical displacement
   measurements. A minimally perturbing fiber-optic probe is another option.
3. Measure the receiver chain with the actual termination, gain, filters, and
   ADC counts-per-volt scale.
4. A single pulse-echo reference identifies only the product of transmit,
   propagation, receive, and electronics. It cannot identify separate Pa/V and
   V/Pa sensitivities without a hydrophone or a justified reciprocity method.
5. Validate linear voltage scaling at low pressure before extrapolating. Stop
   extrapolating when harmonic content, waveform shape, or focal position
   changes with drive.

### 4. Use the model to set geometry, not the ejection threshold

For this 4.22 mm, 80% DMSO case, center a low-drive mechanical water-gap scan
around the nominal 19.1 mm optimum rather than the current 25.3 mm gap. Use
small steps near the maximum and compare the independently mapped incident
field with the calculated curve. Recalculate for every measured
temperature and fill height. In the present model, changing height by ±0.10 mm
moves the optimum by about ∓0.11 mm, while 20–40 °C moves it by about 0.31 mm;
unmeasured PP/probe properties add further uncertainty.

Use the center-frequency focus for alignment, then confirm with the actual
broadband drive. A one-cycle pulse has a broad spectrum and the PP phase is
frequency dependent. At a pressure-release liquid–air boundary, total
first-order pressure is nearly zero. ADE-relevant candidates are therefore the
calibrated **incident** momentum-flux/radiation-stress impulse and total normal
surface velocity, not an undifferentiated “surface pressure.” Pulse-echo
surface amplitude is not automatically an incident-pressure proxy and should
be used only after low-drive correlation to the independent field map.

### 5. Build an empirical ejection map

After acoustic calibration, sweep tone length and terminal voltage as separate
variables. A useful initial sequence is 1, 2, 4, 8, 16, and 32 cycles, followed
by denser sampling near 40–64 cycles because the nominal cavity return begins
near 51.7 cycles. Independently vary and record PRF/inter-shot interval and
duty cycle. Start only after electrical ratings establish a safe lower drive,
then use small controlled steps. Randomize or interleave conditions to reveal
heating and concentration drift.

For each shot, record:

- no ejection / single drop / satellites / spray;
- drop diameter or volume and coefficient of variation;
- launch velocity and angle from high-speed video;
- meniscus deformation and recovery time;
- terminal voltage, current, and net electrical load energy;
- calibrated incident pressure/momentum-flux proxy and total normal velocity;
- temperature and liquid height before and after the shot;
- viscosity, surface tension, wetting state, and degassing protocol;
- PRF, inter-shot interval, duty cycle, and surface recovery state;
- broadband acoustic emissions indicative of cavitation.

Focused-ultrasound experiments show that jet breakup, capillary waves, and
cavitation can change regimes with drive
([Tomita](https://doi.org/10.1063/1.4895902)). Other nozzle-free ejectors show
that acoustic streaming and surface tension can be decisive rather than a
single pressure threshold
([Ning et al.](https://doi.org/10.1039/D1AN01028J),
[Connacher et al.](https://doi.org/10.1103/PhysRevLett.125.184504)). This is why
surface imaging and passive cavitation monitoring are needed alongside the
linear model.

Published focused transducers also show that frequency, pulse width, and drive
amplitude all affect drop size
([Liang et al.](https://doi.org/10.1109/TUFFC.2021.3059904)). Modern devices
use programmable empirical operating maps and assess cell integrity rather
than deriving performance from voltage alone
([Zhang et al.](https://doi.org/10.1038/s41378-024-00798-y)).

## Decision gates before claiming predictive ADE performance

1. **Timing gate:** all three interface timings agree across multiple known
   heights and temperatures without echo-specific fitting.
2. **Field gate:** hydrophone axial/lateral maps agree in focus, width, and
   relative sidelobes at several low voltages.
3. **Electrical gate:** terminal voltage/current and receive loading are known;
   “150 V” has one unambiguous reference plane.
4. **Amplitude gate:** complex Pa/V and receiver/ADC responses are traceable,
   with uncertainty bounds.
5. **Nonlinearity gate:** harmonics, cavitation, and temperature remain inside
   the chosen operating regime.
6. **Ejection gate:** a repeatable measured window exists between first-drop
   threshold and satellite/spray threshold.

Only after these gates should the repository's linear field be coupled to a
transient axisymmetric free-surface solver **if the measured geometry and
wetting are genuinely axisymmetric** (compressible acoustics or a
validated radiation-stress boundary condition plus Navier–Stokes, gravity,
viscosity, and surface tension) to predict mound growth and pinch-off.
