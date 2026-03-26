# DLS-SUC with Modifications
This repository contains a modified implementation of the DLS-SUC model, adapted for practical use on new datasets. The codebase has been restructured to support end-to-end inference, enabling users to input raw biological sequence data and obtain actionable outputs without requiring manual intervention.

The pipeline performs three core functions. First, it preprocesses input data by formatting protein sequences, encoding features, and ensuring compatibility with the trained model. Second, it applies the modified inference engine to generate prediction scores for potential succinylation sites. Third, it produces interpretable outputs, including probability estimates for each candidate site and a ranked list of the top ten most likely succinylation sites.

These modifications transform the original research code into a more usable tool, streamlining data preparation and standardizing outputs for downstream analysis or integration into broader workflows.
