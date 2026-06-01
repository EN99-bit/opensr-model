# Udfordringer og Fejl i Projektet

Denne fil dokumenterer de væsentligste tekniske udfordringer vi stødte på
under udviklingen af S1+S2 fusion super-resolution modellen, hvordan de
manifesterede sig, hvordan de blev diagnosticeret, og hvordan de blev løst.

---

## 1. VAE Weight Loading — Nul vægte loadet (stille fejl)

### Symptom
UNet-denoiseren producerede **ren noise** som output. Ingen struktur, ingen
farver — bare tilfældigt støj. Modellen så ud til at ignorere al conditioning.

### Årsag
VAE-checkpointet (fra `LitVAE` Lightning-modulet) gemte vægte under `vae.*`
prefix, f.eks. `vae.encoder.conv_in.weight`. Men `_load_vae()` i
`train_unet.py` fjernede kun `module.*` prefix. Med `strict=False` i
`load_state_dict()` fejlede det **stille** — ingen fejlmeddelelse, ingen
warning. Resultatet: **0 af 204 VAE-nøgler blev loadet**, og VAE'en kørte
med tilfældige (uinitialiserede) vægte.

Med random VAE-vægte var latent space meningsløst → UNet'en trænede i et
kaotisk rum → inference producerede ren noise.

### Diagnose
Skrev et diagnostik-script der sammenlignede checkpoint-nøgler med de
forventede nøgler i `ldm.first_stage_model`:

```
Expected keys:  204
Checkpoint keys: 264
Matched keys:   0
>>> ZERO MATCH — VAE weights are NOT being loaded!
After removing vae. prefix: 204 matched
```

### Løsning
Omskrev `_load_vae()` til at:
1. Strippe `vae.` prefix fra checkpoint-nøgler
2. Filtrere `disc.*` og `lpips_fn.*` nøgler (discriminator/LPIPS, ikke del af VAE)
3. Bruge `strict=True` så mismatches fejler **højlydt**
4. Printe match-statistik for verifikation

**Resultat**: `VAE loaded: 204 keys (missing: 0, unexpected: 0)` ✅

### Læring
`strict=False` i `load_state_dict()` er farligt — det skjuler fejl. Brug
altid `strict=True` og verificer antal matchede nøgler ved weight loading.

---

## 2. Latent Scaling Factor — Signal dominerer noise

### Symptom
UNet-denoiseren hallucinerede: den genererede bygninger og veje i
skovområder, og output lignede ikke S2-inputtet. Problemet forværredes med
stigende epochs — fra epoch 6-7 begyndte den at "tegne streger" der ikke
fandtes i ground truth.

### Årsag
VAE'ens latent space havde en **standardafvigelse på ca. 5,5** i stedet for
den forventede ≈ 1,0. Diffusionens noise schedule (`linear_start=0.0001`,
`linear_end=0.01`) er designet til at operere på data med std≈1.0.

Med std≈5.5 dominerede signalet over noise ved næsten alle timesteps:

```
Ved t=500 (burde være halvt signal, halvt noise):
  Signal-bidrag:  0.71 × 5.5 = 3.9
  Noise-bidrag:   0.71 × 1.0 = 0.7
  → Signal er 5.6× stærkere end noise!
```

UNet'en lærte aldrig rigtig denoising fra ren noise, fordi den aldrig så
ren noise under træning. Ved inference — hvor vi starter fra ren noise
(std=1.0) — vidste den ikke hvad den skulle gøre og hallucinerede.

### Diagnose
Kørte latent space statistik over 50 tiles:

```
Latent z over 50 samples:
  mean of means: 0.0376
  std of means:  0.4374
  mean of stds:  5.5183    ← FORVENTET ~1.0
```

### Løsning
Tilføjede `scale_factor=0.18215` til `LatentDiffusion`-initialisering i
**både** `train_unet.py` og `srmodel.py`. Værdien 0.18215 ≈ 1/5.5
normaliserer latent-værdier til std≈1.0.

Mekanismen eksisterede allerede i koden (`get_first_stage_encoding()` og
`decode_first_stage()`), men var sat til `scale_factor=1.0` (ingen effekt).

**Resultat**: Signal og noise matcher ved alle timesteps. UNet'en lærer
korrekt denoising. Ingen hallucination. ✅

### Læring
Altid verificer latent space-statistik efter VAE-træning. Hvis std afviger
markant fra 1.0, skal der kompenseres med en scaling factor. Det er
standardpraksis (Stable Diffusion bruger præcis 0.18215).

---

## 3. S2 Båndrækkefølge — BGR vs RGB mismatch

### Symptom
Røde tage på bygninger blev gengivet som grå/blålige i SR-output. Farverne
var generelt "forkerte" men ikke totalt ødelagte.

### Årsag
S2-data blev loadet i **BGR-rækkefølge** (`S2_KEYS = ["s2_b", "s2_g", "s2_r",
"s2_nir"]`), mens aerial-data blev loadet i **RGB-rækkefølge** (`AERIAL_KEYS =
["aerial_r", "aerial_g", "aerial_b", "aerial_nir"]`).

Det betød at kanal 0 var **blå** for S2 men **rød** for aerial. VAE'en var
trænet på aerial (RGB), og S2-conditioning gik også gennem den samme VAE —
men med forkert båndrækkefølge. UNet'en lærte en forkert farve-mapping.

### Diagnose
Verificeret med et simpelt mean-check:

```
S2 order in data.py: s2_b, s2_g, s2_r, s2_nir = BGR
Aerial order:        aerial_r, aerial_g, aerial_b = RGB
MISMATCH: channel 0 is Blue in S2 but Red in Aerial
```

### Løsning
Ændrede `S2_KEYS` i `data.py` til RGB-rækkefølge:

```python
# FRA (forkert):
S2_KEYS = ["s2_b", "s2_g", "s2_r", "s2_nir"]

# TIL (korrekt):
S2_KEYS = ["s2_r", "s2_g", "s2_b", "s2_nir"]
```

Normaliserings-funktionen `normalize_s2()` bruger samme divisor (3000) for
alle RGB-bånd, så ændringen påvirkede ikke normaliseringen.

**Resultat**: Korrekte farver i SR-output. Røde tage er røde. ✅

### Læring
Tjek altid båndrækkefølge mellem datasources. NPZ-filer fra pipelines kan
have anden rækkefølge end hvad modellen forventer.

---

## 4. CFG Dropout under Træning + Manglende CFG ved Inference

### Symptom
SR-output ved `guidance_scale=1.0` (ingen CFG) var **blødt og manglede
detaljer**. Ved `guidance_scale=7.5` blev det skarpere men med alvorlig
**hallucination** (bygninger i skovområder). Ingen guidance-værdi gav
tilfredsstillende resultater.

### Årsag
UNet'en var trænet med `cfg_dropout=0.15` — conditioning blev sat til nul
15% af tiden. Det lærte en "unconditional mode" der genererede
gennemsnitligt dansk landskab (overvejende bebyggelse, da datasættet havde
overvægt af det).

Ved `guidance_scale=1.0`: Modellen brugte sin svage conditional pathway
uden CFG-forstærkning → blødt output.

Ved `guidance_scale=7.5`: CFG-forstærkningen amplificerede forskellen
mellem conditional og unconditional prediction 7.5× → overdreven
detail-generation → hallucination.

### Diagnose
Hyperparametre fra checkpoint bekræftede `cfg_dropout: 0.15`. Test med
forskellige guidance-scales viste at ingen værdi gav gode resultater med
den cfg_dropout-trænede model.

### Løsning
Gentrænede UNet med `cfg_dropout=0.0` — conditioning altid til stede.
LDSR-S2 papiret bruger heller ikke CFG. Ved inference bruges
`guidance_scale=1.0` (standard DDIM, ingen dobbelt UNet-pass).

**Resultat**: Modellen udnytter S1+S2 conditioning fuldt ud. Ingen
hallucination. Ingen krykker. ✅

### Læring
CFG er designet til text-to-image (Stable Diffusion) hvor prompt-signalet
er svagt. For super-resolution er conditioning (S2+S1) det **primære**
input — det skal altid være til stede. CFG er unødvendig og skadelig i
denne kontekst.

---

## 5. CUDA Out of Memory — VAE træning på 10.75 GB GPU

### Symptom
VAE-træningen crashede med `CUDA out of memory` ved `batch_size=8` og
selv ved `batch_size=2` med den fulde loss (VAE + Discriminator + LPIPS/VGG).

### Årsag
GPU'en havde kun 10.75 GB VRAM. Med VAE (55M params), PatchGAN
discriminator (2.8M), LPIPS/VGG backbone (~14M) og alle aktiverings-tensors
fra et 256×256 forward+backward pass var hukommelsen utilstrækkelig.

### Løsning
1. **Gradient checkpointing** i autoencoder Encoder og Decoder — sparer
   ~30-40% VRAM ved at smide aktiverings-tensors væk og genberegne dem
   under backward pass.
2. **Mixed precision** (`--precision 16-mixed`) — halverer hukommelsesforbrug
   for de fleste operationer.
3. **Reduceret batch size** til 2-4.

**Resultat**: VAE-træning kører succesfuldt på 10.75 GB GPU. ✅

### Læring
Gradient checkpointing er essentielt for store modeller på begrænset
hardware. Det koster ~20-30% ekstra compute-tid men muliggør træning der
ellers ville være umulig.

---

## 6. `_tensor_decode()` slettet ved uheld

### Symptom
`SRLatentDiffusion.forward()` kaldte `self._tensor_decode()` som ikke
eksisterede → `AttributeError` ved inference.

### Årsag
Under en whitespace-oprydning af `srmodel.py` blev metoden
`_tensor_decode()` ved et uheld slettet. Da filen havde mange ændringer
(S1+S2 fusion adaptation), gik det ubemærket.

### Diagnose
`grep` for `def _tensor_decode` viste ingen resultater, mens `grep` for
`_tensor_decode(` viste 3 kaldsteder. Metoden blev fundet i git-historikken.

### Løsning
Genskabt metoden med opdateret logik:
- `normalize_aerial(denorm)` i stedet for den gamle `linear_transform`
- `self._X_s2` reference i stedet for `self._X`
- Channel-count safety i histogram matching

**Resultat**: Inference fungerer korrekt. ✅

### Læring
Kør altid en smoke-test (end-to-end forward pass) efter store
refactorings, også selvom ændringerne "bare er whitespace".

---

## 7. Server Environment — Manglende dependencies

### Symptom
Diverse `ModuleNotFoundError` og `CUDA init` fejl ved første kørsel
på serveren.

### Årsag
Flere sammenhængende problemer:
1. System-Python (`python3`) havde ikke torch installeret — `.venv` var
   ikke aktiveret.
2. `pip install torch` installerede CPU-versionen i stedet for CUDA.
3. `tensorboard` manglede i `requirements.txt`.
4. `.venv` fra Windows (kopieret via `scp`) virkede ikke på Linux.

### Løsning
1. Oprettede ny `.venv` direkte på serveren
2. Installerede torch med CUDA: `pip install torch --index-url .../cu121`
3. Installerede manglende `tensorboard`
4. Brugte `rsync` med excludes i stedet for `scp` ved fremtidige uploads

**Resultat**: Fuldt funktionelt træningsmiljø på serveren. ✅

### Læring
Opret altid virtual environment direkte på target-maskinen. Kopiér aldrig
`.venv` mellem OS'er. Brug `rsync --exclude .venv` til deployment.

---

## Opsummering

| # | Fejl | Symptom | Root Cause | Løsning |
|---|---|---|---|---|
| 1 | VAE weight loading | Ren noise output | `vae.` prefix ikke strippet, `strict=False` | Strip prefix, `strict=True` |
| 2 | Latent scaling | Hallucination fra epoch 6-7 | Latent std=5.5, noise schedule antager 1.0 | `scale_factor=0.18215` |
| 3 | S2 båndrækkefølge | Forkerte farver (rødt→gråt) | BGR vs RGB mismatch i data loading | Ændr S2_KEYS til RGB |
| 4 | CFG dropout | Blødt eller hallucineret output | 15% conditioning dropout + ingen CFG inference | Retræn med cfg_dropout=0 |
| 5 | CUDA OOM | Træning crasher | 10.75 GB GPU, fuld loss pipeline | Gradient checkpointing + mixed precision |
| 6 | Slettet metode | AttributeError ved inference | Whitespace cleanup slettede _tensor_decode | Genskabt fra git |
| 7 | Server env | Import errors, CUDA init fejl | Forkert Python, CPU torch, manglende deps | Ny venv, CUDA torch, tensorboard |
