# Fluid–Elastic–Solid–Fluid Angular-Spectrum-Modell

Lokale Python-Implementierung für den im geteilten Chat beschriebenen Aufbau:

| Größe | Wert im Beispiel |
|---|---:|
| Mittenfrequenz | 10 MHz |
| Transducer-Apertur | 13 mm |
| geometrischer Fokus | 25 mm |
| Temperatur | 22 °C |
| Schichten | Wasser → PP → DMSO |
| PP-Dicke | 0,78 mm |
| gemessene longitudinale PP-Schallgeschwindigkeit | **2732 m/s** |

Das Modell ist kein paraxiales Fresnel-Modell. Es propagiert jede FFT-Komponente
mit

\[
k_z = \sqrt{k^2-k_x^2-k_y^2}
\]

und löst an der PP-Platte für jede transversale Wellenzahl ein vollständiges
Fluid–Festkörper–Fluid-Randwertproblem. Im isotropen PP werden vorwärts und
rückwärts laufende Longitudinal- und SV-Wellen berücksichtigt. Dadurch sind
Modenkonversion, winkelabhängige Transmission und sämtliche
Mehrfachreflexionen in der Platte enthalten.

Die Vorgehensweise folgt dem etablierten Angular-Spectrum-Ansatz für
geschichtete Medien ([Vecchio et al., 1994](https://pubmed.ncbi.nlm.nih.gov/7810021/))
und der Potentialmethode für fluidgekoppelte elastische Platten
([Almeida et al., 2023](https://arxiv.org/abs/2302.08826)). Der Einfluss des
endlichen Winkelspektrums auf Plattenresonanzen ist experimentell und numerisch
dokumentiert ([Aanes et al., 2016](https://arxiv.org/abs/1604.02258));
eine verwandte offene ASM-Implementierung für einen Kolbenstrahler findet sich
bei [Sæther, 2023](https://doi.org/10.1016/j.mex.2023.102037).

## Schnellstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
asm-pp-case
```

Die Standardrechnung erzeugt im Ordner `results/`:

- `axial_scan.png`: axialer Druck mit PP und eine Referenz ohne PP
- `focal_plane.png`: 2D-Druckfeld in der berechneten Fokalebene
- `lateral_profile.png`: laterales -6-dB-Profil
- `transmission_vs_angle.png`: winkelabhängige Reflexion und Transmission
- `transmission_vs_frequency.png`: Plattenresonanzen
- `monochromatic_results.npz`: komplexe Felder und Kurven
- `summary.json`: Parameter, Fokuslage, FWHM und Plausibilitätswerte

Eine breitbandige, vom Transducer bandpassgefilterte
Drei-Zyklen-Rechteckanregung wird optional rekonstruiert:

```bash
asm-pp-case --pulse
```

Für eine 4,22-mm-Schicht mit 70-100 % DMSO steht ein eigener Sweep zur
Verfügung. Standardmäßig werden Volumenprozent angenommen:

```bash
asm-dmso-sweep --basis volume --dmso-height-mm 4.22
# oder als direkt editierbares Beispiel:
python examples/dmso_concentration_sweep.py
```

Die Mischungseigenschaften werden nicht linear zwischen Wasser und DMSO
gemittelt. Sie werden aus gemessenen Dichte- und Schallgeschwindigkeitsdaten
bei 20 und 40 °C interpoliert
([Palaiologou et al., 2006](https://doi.org/10.1007/s10953-006-9082-5)).

## Wichtige, noch zu messende Parameter

Aus dem Chat ist nur die longitudinale PP-Geschwindigkeit gemessen. Deshalb
sind folgende Standardwerte ausdrücklich **Startwerte**, keine Kalibration:

- PP-Dichte: 900 kg/m³
- PP-Querkontraktionszahl: 0,42; daraus folgt \(c_S \approx 1015\) m/s
- PP-Dämpfung für P und SV: standardmäßig 0 dB/m
- DMSO bei 22 °C: 1499 m/s und 1098,4 kg/m³ als Literatur-Näherung
- Wasser bei 22 °C: 1488,4 m/s und 997,77 kg/m³
- Wasserweg bis zur PP-Vorderseite: 20 mm

Für belastbare absolute Drücke sollten mindestens PP-Dichte,
PP-Scherwellengeschwindigkeit, longitudinale und transversale Dämpfung sowie
Schallgeschwindigkeit/Dichte der tatsächlich verwendeten DMSO-Konzentration
gemessen werden. Ein eigener Messwert wird beispielsweise so eingesetzt:

```bash
asm-pp-case \
  --water-path-mm 20.0 \
  --pp-density 905 \
  --pp-shear-speed 1080 \
  --pp-alpha-l-db-m 2500 \
  --pp-alpha-s-db-m 4000 \
  --dmso-speed 1497 \
  --dmso-density 1097
```

Die Dämpfungswerte sind Amplitudenverluste in dB/m bei der simulierten
Mittenfrequenz. Ohne gemessene Dämpfung kann die Resonanzlage sinnvoll sein,
die Resonanzhöhe und damit der Spitzendruck aber nicht.

## Verwendung als Python-Bibliothek

```python
import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
)

water = Fluid("water_22C", 997.77, 1488.4)
dmso = Fluid("DMSO_22C", 1098.4, 1499.0)
pp = ElasticSolid.from_longitudinal_speed_and_poisson(
    name="PP",
    density_kg_m3=900.0,
    longitudinal_speed_m_s=2732.0,  # gemessen
    poisson_ratio=0.42,              # ersetzen, sobald cS gemessen ist
)

model = AngularSpectrumModel(
    grid=CartesianGrid(nx=512, ny=512, dx_m=40e-6),
    aperture=FocusedCircularAperture(13e-3, 25e-3),
    incident_fluid=water,
    plate=ElasticPlate(pp, 0.78e-3),
    transmitted_fluid=dmso,
    water_path_m=20e-3,
)

z_after_pp = np.linspace(0.0, 10e-3, 161)
axis_pressure = model.on_axis_scan_after_plate(10e6, z_after_pp)
focus_index = np.argmax(np.abs(axis_pressure))
focus_z = 20e-3 + 0.78e-3 + z_after_pp[focus_index]
focal_plane = model.field_after_plate(10e6, z_after_pp[focus_index])
print(f"Fokus bei {focus_z * 1e3:.3f} mm")
```

Ein vollständig editierbares Beispiel liegt unter
[`examples/custom_case.py`](examples/custom_case.py).

## Physikalischer Umfang und Grenzen

Enthalten:

- exakte 3D-FFT-Propagation ohne paraxiale \(k_z\)-Näherung
- propagierende und evaneszente Komponenten mit auslaufendem Wurzelzweig
- P-/SV-Modenkonversion in einer isotropen elastischen PP-Platte
- Mehrfachreflexionen und Plattenresonanzen
- frequenz- und winkelabhängige komplexe Transmission
- verlustbehaftete Medien über komplexe Wellenzahlen
- optional bandbegrenzte ASM-Auswertung gegen numerisches Wrap-around
- optionale Zeitrekonstruktion eines breitbandigen Pulses

Nicht enthalten:

- elastische Anisotropie einer orientierten/extrudierten PP-Folie
- endliche seitliche Abmessungen oder Verkippung der PP-Platte
- Oberflächenrauheit, Verklebungen, Luftblasen oder zusätzliche Schichten
- Nichtlinearität bei hohen Schalldrücken
- ein elektromechanisches Transducer-Ersatzschaltbild

Die Quelle ist eine planare äquivalente Kreisapertur mit exakter sphärischer
Fokussierungsphase. Der Standardwert von 1 Pa an der Apertur liefert daher
einen relativen Fokussierungsgewinn. Für absolute Hydrophone-Drücke muss
`pressure_amplitude_pa` oder ein gemessener komplexer
Spannung-zu-Druck-Frequenzgang eingesetzt werden.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Die Tests prüfen insbesondere:

- den auslaufenden/abklingenden \(k_z\)-Zweig
- Übereinstimmung mit der analytischen Schichtformel bei Normalinzidenz
- Energieerhaltung \(R+T=1\) für eine verlustfreie Platte
- Fokus und endliche Felder auf einem reduzierten Gitter
- die Zeitkonvention der Pulsrekonstruktion
