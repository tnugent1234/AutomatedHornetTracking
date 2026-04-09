# Automated Hornet Tracking
An automated system for quantifying invasive hornet activity at colonies via deep learning and markerless tracking. This enables the analysis of behaviour in nature from remote video footage, including foraging rates, spatial activity, and individual trajectory information. The system is designed to detect and track the hornet *Vespa velutina nigrithorax*, with additional functionality for *V. bicolor*, *V. orientalis*, and *V. tropica*.

<p align="center"><img src=https://github.com/tnugent1234/AutomatedHornetTracking/blob/main/Images/Hornet%20Tracking%20Examples.gif/></p>

## Contents
* [Models](Models) Trained YOLO11n detection models for use with the system.
* [Scripts](Scripts) Central processing pipeline and GUI scripts for running the system.
* [Trackers](Trackers) ByteTrack and CustomTrack .yaml files for configuring tracking algorithms.
* [Images](Images) Example tracking sequences from the system.

## Install
Download the repository into your Python directory or virtual environment, and run [requirements.txt](requirements.txt) to install all required packages.

## Quick Start
Once installed, run [HornetTrackerGUI.py](HornetTrackerGUI.py) to use the system.
