# Simulation-based Star Formation History Inference Hackathon
[![forthebadge](https://forthebadge.com/images/badges/uses-badges.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/built-with-science.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

Repository for SFH inference hackathon at [AstroInfo 2021](https://astroinfo2021.sciencesconf.org/).

Proposed by: [Marc Huertas-Company](https://github.com/mhuertascompany), [Francois Lanusse](https://github.com/eiffl), [Alexandre Boucaud](https://github.com/aboucaud)

![image](https://user-images.githubusercontent.com/861591/144759151-1091c201-2cb0-433e-aa81-6c8728afc579.png)


See [this issue](https://github.com/EiffL/sfh-inference-hackathon/issues/1) to get started.

## How to get started on Jean-Zay

1. Log on the machine

2. Clone this repo
```bash
git clone https://github.com/EiffL/sfh-inference-hackathon.git
```

3. Load the environment
```bash
module load tensorflow-gpu/py3/2.6.0
```

4. Start an interactive session
```bash
srun --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --time=06:00:00 -A qrc@gpu --reservation=hackathon --pty bash
```

5. Start a jupyter lab
```bash
idrlab --notebook-dir=$PWD
```
And then follow the link to https://idrvprox.idris.fr, log in, and click on your submitted job.

Then you can try to load the [intro notebook](notebooks/Intro_Hackathon_Astroinfo21_SFHs.ipynb)
