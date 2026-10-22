# MMA3001 - Occupancy Prediction

## Project Overview

This project investigates machine learning methods for predicting building occupancy from temporal occupancy patterns.

## Objective

Train and compare machine learning models using occupancy data from Building 60, Room G.20, and evaluate their ability to generalise to Room G.25.

## Dataset

The dataset contains approximately 9 million occupancy records across five floor spaces.

The original datasets are stored in Data/ and tracked using Git LFS.

## Repository Structure

- Data/: Original datasets and metadata
- src/: Main occupancy prediction scripts
- archive/: Previous exploratory scripts
- tests/: Model and preprocessing tests
- results/: Figures and evaluation results
- docs/: Project documentation

## Planned Workflow

1. Extract occupancy data for G.20 and G.25.
2. Investigate headcount and occupancy status distributions.
3. Convert event records into hourly occupancy features and targets.
4. Train and compare linear and nonlinear models.
5. Test model generalisation on the unseen room.
6. Evaluate accuracy, limitations and sensitivity.

## Unit

MMA3001 - Numerical Methods and Machine Learning
Monash University
