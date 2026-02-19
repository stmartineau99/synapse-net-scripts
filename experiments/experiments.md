# Deepict actin segmentation models
Scripts used to perform each experiment are kept in separate folders.

## Synthetic data preparation

### Deepict dataset 1
60 tomograms were generated using PolNet, and noise was added with Faket style transfer using an SNR range of 0.4 to 0.5. 

Three conditions with 20 tomograms each were used to simulate data:
1. membranes disabled, actin pmer occ 0.7%, mt pmer occ 0.3%
2. membranes disabled, actin pmer occ 0.3%, mt pmer occ 0.7%
3. membranes enabled, actin pmer occ 0.25%, mt pmer occ 0.15%

## Model training
- **RUN 1** supervised training on synthetic data only (60 tomograms)
- **RUN 2** increase lr from 1e-4 to 4e-4 to account for increase in batch size from 1 to 4 (linear scaling).
- **RUN 3** Domain adapt model from 1. to deepict dataset (3 tomograms) 
- **RUN 4** Implement clDice and repeat 1. and 2.
- **RUN 5** Classification model on synthetic data with actin, microtubules, and membranes.
- **RUN 6** Domain adapt model from 4. to deepict dataset, and punish the model for segmenting MT and membranes.

These experiments can be repeated for the optogenetics dataset.
