# Deepict actin segmentation models
Scripts used to perform each experiment are kept in separate folders.

## Synthetic data preparation
60 tomograms were generated using PolNet, and noise was added with Faket style transfer using an SNR range of 0.4 to 0.5. 

Three conditions with 20 tomograms each were used to simulate data:
1. membranes disabled, actin pmer occ 0.7%, mt pmer occ 0.3%
2. membranes disabled, actin pmer occ 0.3%, mt pmer occ 0.7%
3. membranes enabled, actin pmer occ 0.25%, mt pmer occ 0.15%

## Experiments
**RUN 1**supervised training on synthetic data only (60 tomograms)
**RUN 2** Domain adapt model from 1. to deepict dataset (3 tomograms)
**RUN 3** Implement clDice and repeat 1. and 2.
**RUN 4** Classification model on synthetic data with actin, microtubules, and membranes. 
**RUN 5** Domain adapt model from 4. to deepict dataset, and punish the model for segmenting MT and membranes

These experiments can be repeated for the optogenetics dataset.