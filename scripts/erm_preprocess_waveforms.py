#!/usr/bin/env python3
"""
Preprocess raw uint8 haptic arrays into ERM-ready Arduino arrays.

Default usage from the repository root:

  1. Paste your raw sequence(s) into PASTED_STIMULI below.
  2. Run:

     python scripts/erm_preprocess_waveforms.py

The script does not edit or read the .ino file by default. It processes pasted
raw arrays offline and writes a paste-ready Arduino fragment.

By default it converts 10 ms / 200-frame / 2 s raw sequences into
20 ms / 100-frame / 2 s ERM-ready sequences. The 10 ms -> 20 ms conversion
uses peak-hold max pooling by default, so short vibration peaks are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


# Default preprocessing parameters. Override any of these from the CLI.
THRESHOLD = 20
GAMMA = 0.65
MIN_NONZERO = 35
MAX_DRIVE = 220
MIN_PULSE_FRAMES = 3
ATTACK_FRAMES = 1
RELEASE_FRAMES = 2
INPUT_SAMPLE_INTERVAL_MS = 10
OUTPUT_SAMPLE_INTERVAL_MS = 20
MAX_FRAMES = 200


# Paste raw data here, then run:
#
#   python scripts/erm_preprocess_waveforms.py
#
# Accepted paste formats:
#
# 1. One raw comma-separated sequence:
#      0, 5, 12, 25, 40, 80, 120, 40, 0
#
# 2. One or more Arduino arrays:
#      const uint8_t STIMULUS_01[] = {0,5,12,25,40};
#      const uint8_t STIMULUS_02[] = {251,249,246,255};
#
# If you paste only one unnamed sequence, the output array name will be
# STIMULUS_01 so it can be pasted directly into the Arduino sketch.
PASTED_STIMULI = r"""
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SUCCESS

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_04[] = {0,0,26,41,46,64,49,53,53,49,63,52,54,52,53,90,49,56,57,51,57,54,54,55,53,52,54,55,52,53,55,54,53,51,50,45,40,36,27,0,0,7,7,8,11,10,7,12,22,26,25,27,22,18,23,29,24,37,62,68,65,94,91,82,74,70,80,82,77,73,77,76,200,200,63,54,46,45,42,41,38,36,36,30,30,28,28,25,18,20,15,8,20,19,19,19,20,20,18,17,19,17,17,18,18,14,17,15,12,13,12,14,14,14,14,12,10,10,11,12,11,10,11,8,9,7,8,9,10,0,0,5,9,16,23,73,87,86,49,32,35,25,23,16,14,13,13,11,12,10,9,8,255,10,11,8,7,6,6,7,8,10,42,66,61,81,89,66,59,57,40,31,20,15,28,22,14,14,16,12,14,14,17,14,12,12,13,11,13,10,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_05[] = {56,61,69,68,64,43,23,18,39,51,47,53,56,50,40,40,36,28,34,149,255,32,36,46,47,48,41,42,47,46,40,43,44,43,46,46,38,35,41,34,23,33,40,46,55,51,54,48,50,53,52,50,50,50,21,0,0,6,8,6,8,8,8,11,12,17,22,23,16,16,19,15,8,7,8,6,6,6,6,5,6,7,6,7,6,7,5,7,5,7,6,6,6,5,6,7,6,7,6,7,8,9,11,12,11,9,188,13,14,22,23,18,26,25,27,26,23,30,21,24,21,18,14,16,22,20,14,13,10,11,8,9,11,10,7,8,6,6,8,8,5,7,7,8,6,6,8,9,5,6,6,6,7,6,7,5,6,6,6,6,6,6,6,5,6,6,6,7,6,6,6,5,7,7,6,6,6,7,7,6,8,6,7,6,6,6,5,6,6,6,7,6,6,4,0,0,0,0,0,0};
const uint8_t STIMULUS_06[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,10,8,14,32,33,30,33,56,54,44,30,25,25,23,16,13,12,12,11,13,12,13,12,14,12,11,15,11,8,10,8,8,8,8,5,7,8,8,7,9,7,5,255,9,6,8,7,10,9,7,11,9,6,7,6,7,6,7,8,5,9,11,8,9,9,7,8,7,9,12,9,7,7,6,4,8,7,6,5,6,6,6,5,5,6,6,6,6,5,5,6,6,7,5,5,6,6,6,7,8,8,6,4,5,7,7,9,6,7,10,12,13,10,23,20,22,21,15,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


// MY VAE
const uint8_t STIMULUS_13[] = {60,59,69,66,75,79,82,84,84,86,115,80,86,83,87,81,82,82,75,77,75,71,79,50,64,62,39,51,47,46,37,38,41,51,60,62,131,59,59,59,63,69,66,67,67,66,59,60,63,68,75,70,65,69,71,68,69,72,72,70,68,73,73,75,72,76,72,75,80,77,70,74,72,72,75,73,76,78,81,76,81,75,75,75,77,75,78,78,76,78,73,69,64,61,64,60,60,59,65,43,2,0,0,0,0,0,0,0,3,11,38,56,45,18,8,7,7,8,7,7,9,9,13,255,11,9,13,11,8,8,7,8,7,7,9,7,9,7,7,6,7,7,7,7,7,7,7,7,7,6,0,0,0,0,0,0,0,0,5,14,13,17,17,16,28,34,49,55,52,36,36,29,22,221,16,15,16,11,12,10,11,11,11,12,10,11,10,13,13,13,22,32,43,43,34,24,22,19,16,17};
const uint8_t STIMULUS_14[] = {78,82,89,88,97,102,105,111,108,109,157,104,107,103,107,102,102,100,92,98,92,92,108,75,83,79,59,72,71,65,56,53,71,70,76,71,109,70,80,72,83,81,79,82,79,76,74,77,80,83,94,83,80,82,87,81,87,88,89,87,85,92,88,94,93,102,99,101,108,104,97,97,99,97,106,98,100,104,105,105,106,99,97,95,94,91,97,100,97,99,92,87,84,83,85,75,77,75,80,56,1,0,0,0,0,0,0,0,3,13,21,33,18,12,9,9,7,10,9,9,15,29,41,255,22,16,19,15,15,14,14,12,9,11,12,8,9,7,9,7,7,8,8,8,7,8,8,8,8,8,0,0,0,0,0,0,0,0,6,15,58,69,46,16,12,13,10,10,9,9,9,10,14,254,12,10,9,10,10,10,10,11,9,14,10,15,10,10,8,8,8,10,9,9,9,10,10,10,10,7};
const uint8_t STIMULUS_15[] = {0,0,19,26,24,24,26,26,24,27,24,27,28,25,29,25,24,25,25,27,20,28,28,25,28,28,26,34,30,26,31,33,27,30,164,39,29,25,23,27,15,31,27,23,26,37,25,26,34,33,32,31,39,37,38,42,40,41,45,45,43,38,39,42,39,35,28,34,25,29,29,27,16,17,9,20,18,22,32,35,37,34,35,32,33,34,31,36,32,23,2,0,9,11,46,54,78,66,45,53,61,96,97,88,50,22,56,75,74,64,43,77,87,245,101,84,72,49,46,50,48,46,48,45,41,41,39,35,31,24,32,28,30,31,33,26,29,24,24,26,27,23,24,17,23,15,17,16,18,10,15,0,0,6,16,14,17,16,16,20,40,70,61,26,31,25,255,16,16,17,14,12,11,12,12,13,11,10,11,13,16,18,37,56,52,34,19,20,23,15,0,0,0,0,0,0,0,0,0,0};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Error block: anonymous IDs 16-30

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_19[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,35,38,40,37,38,32,38,35,42,33,41,43,37,41,40,37,32,37,41,37,43,41,52,49,47,46,50,47,255,50,28,20,11,18,24,31,41,51,40,59,60,63,57,52,50,50,52,47,54,53,49,50,44,51,45,40,46,44,41,39,39,39,41,35,38,38,43,44,42,33,34,30,30,33,33,36,37,31,26,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_20[] = {36,90,108,92,118,129,120,125,139,135,131,138,216,122,118,129,131,133,122,128,137,121,127,127,126,130,166,116,131,129,123,121,121,129,122,122,121,138,127,119,112,122,71,255,138,107,114,117,110,128,136,137,131,136,131,120,131,131,152,139,161,151,149,148,144,146,138,144,143,145,143,141,139,149,157,167,162,152,165,154,164,174,157,161,149,150,151,148,155,152,155,154,151,154,152,151,158,147,151,147,145,145,137,142,141,129,132,140,124,116,121,107,65,44,50,74,88,98,63,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_21[] = {18,38,55,62,81,72,69,54,36,24,45,52,43,34,24,35,59,65,62,66,66,43,52,55,52,169,47,44,34,32,30,32,30,28,29,26,27,26,23,20,23,20,18,17,17,17,16,18,21,18,18,17,18,16,14,15,12,15,16,15,13,16,11,16,15,14,11,15,14,0,0,0,0,0,0,0,0,12,21,17,17,16,18,18,19,18,15,18,19,18,22,20,22,119,26,22,22,24,23,26,18,13,11,18,19,23,28,26,25,21,21,19,13,10,16,21,22,21,22,20,22,20,16,0,0,0,0,0,0,0,0,8,10,18,39,56,63,84,97,102,107,122,118,118,137,139,144,126,122,116,94,88,81,92,91,97,255,85,78,67,62,61,54,56,52,54,48,49,43,41,42,35,33,33,30,27,31,29,32,31,33,28,24,23,20,17,22,22,23,21,20,15,11,11,14,10,11,12,15,13};


// MY VAE
const uint8_t STIMULUS_28[] = {5,9,11,22,20,15,9,6,7,7,9,8,8,8,11,12,25,27,226,21,12,17,14,12,13,12,11,10,8,7,9,7,7,9,7,8,7,6,7,7,7,7,7,7,7,7,7,7,7,6,0,0,0,0,0,0,0,0,8,11,9,12,11,13,11,13,12,10,12,11,19,26,25,41,37,39,49,50,36,31,36,29,23,27,20,17,14,255,14,12,14,10,16,14,13,11,12,11,11,9,12,15,11,13,11,14,11,10,12,10,9,12,11,12,12,12,13,16,13,19,28,37,38,52,56,49,42,28,25,20,17,21,13,18,12,22,20,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_29[] = {5,7,20,24,13,8,5,4,4,5,5,5,7,8,14,150,76,12,11,11,8,7,6,5,6,4,5,4,4,5,4,4,4,4,4,4,4,4,4,4,4,4,4,2,0,0,7,7,10,8,9,10,11,9,10,10,16,17,16,22,22,32,34,27,24,24,24,25,17,14,13,142,11,9,10,10,11,10,8,8,9,7,6,8,7,8,7,8,6,6,6,5,6,5,6,7,5,7,9,9,13,22,28,39,44,43,29,25,16,12,11,13,11,14,11,6,0,0,9,16,13,12,46,81,94,73,47,42,41,35,23,25,18,17,13,17,17,13,14,13,14,11,9,255,13,10,11,11,9,9,10,8,7,11,16,45,70,62,69,97,84,54,62,62,40,37,31,20,20,27,23,17,17,16,18,12,16,16,18,12,18,12,13,14,14,16,13,15,0,0,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_30[] = {3,6,7,13,12,10,6,4,5,5,5,6,5,5,7,8,14,16,132,12,8,12,10,9,8,8,7,7,5,5,6,4,4,5,4,5,4,4,4,4,4,4,4,4,5,4,4,4,4,4,0,0,0,0,0,0,0,0,4,6,7,6,7,6,6,7,10,21,26,41,45,31,31,23,14,11,141,9,9,10,9,6,7,7,6,7,8,8,7,8,6,7,7,7,6,8,8,15,17,23,21,11,11,11,10,9,8,9,0,0,0,0,0,0,0,0,9,9,13,11,14,50,80,84,66,47,38,30,28,24,18,21,16,14,11,11,14,11,10,11,10,11,11,9,10,255,9,9,10,7,8,8,7,8,8,8,9,8,29,61,59,23,69,100,90,73,62,76,66,43,37,31,18,17,21,22,14,13,14,13,14,13,13,14,11,16,8,13,10,10,12,13,12,15,13,7,0,0,0,0};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Notification block: anonymous IDs 31-45

//RANDOM VECTOR - VAE
const uint8_t STIMULUS_34[] = {27,66,100,118,112,86,62,95,87,84,64,47,71,235,255,59,46,39,39,37,35,26,21,20,22,20,24,26,23,19,17,20,21,17,13,13,12,13,3,0,4,21,24,23,29,24,23,26,25,26,26,21,25,24,24,26,22,22,30,24,21,27,28,26,27,21,22,21,24,24,20,22,21,21,28,27,27,25,26,28,27,25,24,24,26,31,30,29,29,26,28,29,29,23,24,25,186,36,29,25,27,26,19,20,26,21,25,23,25,22,11,11,14,15,14,8,14,19,27,26,33,31,28,31,28,29,32,24,27,31,30,29,27,31,30,31,26,31,27,31,34,33,30,30,35,35,33,32,31,28,33,35,35,36,32,31,31,33,32,32,27,29,34,27,29,26,26,31,24,26,29,26,23,26,28,25,25,25,26,21,22,23,23,21,23,22,25,21,27,25,22,19,17,17,0,0,0,0,0,0};
const uint8_t STIMULUS_35[] = {9,12,14,13,14,11,11,13,16,30,42,64,71,48,45,40,26,21,255,15,14,18,15,12,12,12,11,12,11,13,10,12,10,12,11,13,13,15,15,33,37,57,57,40,28,22,21,20,20,19,0,0,0,0,0,0,0,0,15,39,35,32,32,34,32,29,27,37,26,36,36,30,34,33,31,37,38,37,37,39,41,48,46,41,41,44,46,208,50,40,33,37,36,32,20,17,20,31,34,51,45,48,42,41,43,41,45,44,42,48,53,45,55,49,46,52,57,56,44,52,42,48,40,39,43,38,38,14,17,21,26,33,28,24,41,39,21,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_36[] = {0,0,8,10,13,14,41,86,90,63,48,36,33,29,26,21,16,14,12,11,13,10,12,11,11,11,8,255,11,8,10,9,8,7,7,6,6,8,7,17,44,39,22,57,60,51,39,47,41,30,28,18,14,20,22,14,13,14,14,12,9,13,11,15,11,14,10,10,12,10,14,12,8,0,1,59,68,69,77,83,85,89,119,90,89,93,94,84,86,85,103,96,87,84,84,80,68,68,74,70,150,73,75,83,86,90,90,85,82,87,91,90,88,85,88,88,89,91,85,89,88,86,90,90,85,86,85,88,88,91,94,91,89,87,90,85,85,83,79,78,69,66,64,58,66,28,0,2,8,9,40,71,34,30,28,17,13,11,11,9,9,8,13,193,8,6,6,6,7,12,46,42,53,47,36,33,23,16,18,16,12,12,10,11,13,12,9,10,11,8,0,0,0,0,0,0,0,0,0,0};

// MY VAE
const uint8_t STIMULUS_43[] = {5,10,14,31,32,16,10,7,7,8,9,8,8,9,11,12,25,26,255,25,16,20,14,13,13,11,12,10,8,8,9,7,7,8,7,7,7,7,8,7,8,8,7,7,7,8,7,7,7,6,0,0,0,0,0,0,0,0,15,28,25,25,23,24,21,23,22,26,22,25,27,23,23,25,22,24,26,26,25,28,26,30,30,27,29,29,25,134,32,23,22,21,16,25,24,19,29,22,16,15,14,10,15,16,25,27,33,34,35,39,40,35,40,40,40,37,34,39,35,31,32,25,25,18,12,9,14,26,27,28,29,32,28,30,31,30,29,20,0,0,0,0,0,0,0,0,9,10,13,12,13,12,13,17,25,35,39,55,55,44,36,36,25,24,244,19,19,22,19,15,17,14,14,13,15,15,11,14,11,13,11,13,13,14,15,23,28,42,47,35,32,26,22,22,22,19,0,0,0,0};
const uint8_t STIMULUS_44[] = {5,9,11,19,21,12,8,6,7,7,8,7,8,9,11,13,26,26,216,21,15,18,13,12,12,11,10,9,8,6,8,6,7,8,6,7,6,6,6,7,7,7,6,6,6,6,6,6,6,5,0,0,0,0,0,0,0,0,5,12,20,67,57,37,13,9,9,9,9,8,8,8,9,9,10,11,255,12,11,13,12,11,10,10,9,11,7,9,12,8,12,13,8,7,8,7,8,7,8,7,8,7,9,8,8,7,9,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_45[] = {6,12,26,49,38,25,10,10,7,7,8,7,8,7,9,9,12,13,255,12,9,14,11,11,9,8,8,9,6,7,8,7,6,7,7,6,6,6,7,7,7,6,7,7,7,7,7,7,7,5,0,0,0,0,0,0,0,0,5,12,30,71,60,44,15,9,9,9,8,9,8,7,8,8,9,9,238,8,9,15,13,10,11,12,9,10,7,9,10,8,10,12,8,7,7,6,7,8,8,8,8,8,9,8,8,8,8,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Refresh/Loading block: anonymous IDs 46-60
//RANDOM VECTOR - VAE
const uint8_t STIMULUS_49[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,10,13,17,16,14,12,16,21,22,42,51,38,32,31,30,21,19,255,16,15,19,16,12,13,11,11,12,12,12,11,9,9,11,11,12,12,14,13,25,34,52,57,43,34,22,21,21,27,17,0,0,0,0,0,0,0,0,9,15,15,13,14,16,14,17,15,15,15,15,18,18,22,21,21,23,99,22,18,19,15,13,22,22,23,19,19,19,21,19,24,23,23,20,22,19,20,19,17,18,16,14,14,13,14,14,15,10,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_50[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,11,15,15,16,15,15,24,30,55,55,70,79,53,55,33,24,17,255,15,16,17,13,12,12,11,11,13,13,14,12,12,10,13,13,13,13,15,25,38,37,51,44,28,24,22,20,19,19,18,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_51[] = {19,47,56,49,59,64,60,63,67,69,68,70,103,65,65,68,69,67,62,64,66,60,63,59,63,61,73,61,61,58,61,49,46,48,47,48,45,46,33,32,41,35,38,118,50,42,46,50,42,53,54,57,52,54,53,50,49,49,55,54,61,58,57,58,57,57,57,59,60,60,59,60,58,62,63,65,64,60,67,62,66,71,66,68,62,64,65,62,67,68,65,66,67,71,68,68,71,68,65,63,66,64,60,64,66,62,63,60,59,57,54,49,51,46,49,48,49,54,39,31,0,0,0,0,0,0,0,0,10,11,15,19,29,30,20,23,10,11,9,8,8,9,8,11,11,10,9,9,11,9,12,17,21,16,255,36,35,36,40,46,40,37,29,35,32,16,17,18,16,14,11,12,9,9,9,13,10,10,11,10,10,8,10,9,9,10,10,9,10,10,9,9,11,9,10,9,9,8,9,8};


// MY VAE
const uint8_t STIMULUS_58[] = {0,0,16,24,26,25,21,23,20,24,21,23,24,22,25,21,21,21,22,23,17,24,25,22,25,24,24,30,26,24,26,28,24,26,142,32,25,25,25,27,23,31,29,30,32,36,32,33,33,35,35,36,35,33,32,38,36,36,36,39,38,37,39,38,38,33,33,35,27,28,29,27,18,17,11,10,10,10,22,26,29,27,27,26,26,27,27,28,25,21,1,0,7,26,58,71,97,116,115,113,85,36,30,70,83,76,84,81,55,17,52,67,72,255,71,62,56,48,40,39,37,38,37,36,28,27,24,25,18,21,21,20,23,22,26,23,20,18,15,18,19,19,17,15,12,13,15,14,13,12,18,1,0,5,9,14,13,7,7,6,7,6,6,8,11,21,150,13,11,10,10,11,9,7,6,7,5,5,5,6,5,5,5,5,5,5,6,5,5,4,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_59[] = {0,0,16,25,28,25,26,23,23,23,24,23,27,23,28,27,23,24,26,25,23,28,28,26,28,29,27,33,29,27,30,32,28,28,137,35,25,20,17,18,18,21,14,11,16,13,26,23,35,32,32,34,31,32,32,35,37,37,35,37,39,38,39,37,38,36,39,39,32,33,31,26,23,24,21,23,24,27,31,29,28,28,27,27,27,28,26,29,27,22,1,0,6,20,47,59,88,103,107,106,90,51,44,75,88,81,83,80,55,34,60,76,79,255,83,68,61,52,44,44,40,41,40,39,31,33,28,29,22,19,23,23,25,24,28,25,21,17,17,19,21,21,17,17,13,14,15,13,12,12,17,0,0,6,12,25,24,14,8,8,9,7,8,9,9,20,209,19,18,13,13,14,12,9,8,10,7,9,7,8,7,7,7,7,7,8,7,7,7,5,0,0,0,0,0,0,0,0,0,0};
const uint8_t STIMULUS_60[] = {0,0,0,0,0,0,0,0,0,0,0,0,14,35,41,35,36,35,37,35,31,33,34,33,28,33,35,28,34,35,32,27,29,28,33,25,30,26,32,32,33,31,32,35,32,30,33,39,39,36,31,31,37,31,29,209,41,43,37,32,30,29,30,34,28,19,25,33,29,24,35,38,41,39,45,40,45,39,38,44,43,43,39,38,36,44,45,47,42,48,48,52,45,53,51,52,52,48,51,49,55,56,48,43,46,48,47,41,39,21,19,18,20,21,14,32,42,39,42,40,37,39,37,36,38,39,40,39,38,34,31,15,0,0,0,0,0,0,0,0,5,11,17,38,31,20,8,7,7,8,8,7,8,8,10,9,20,21,255,18,10,15,12,13,11,10,11,10,6,8,9,7,6,7,7,7,7,6,8,7,7,7,7,7,7,7,7,7,7,5,0,0,0,0,0,0,0,0,0,0};
"""
ARRAY_RE = re.compile(
    r"const\s+uint8_t\s+(STIMULUS_\d{2})\s*\[\]\s*=\s*\{(?P<body>.*?)\};",
    re.DOTALL,
)
INT_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Params:
    threshold: int = THRESHOLD
    gamma: float = GAMMA
    min_nonzero: int = MIN_NONZERO
    max_drive: int = MAX_DRIVE
    min_pulse_frames: int = MIN_PULSE_FRAMES
    attack_frames: int = ATTACK_FRAMES
    release_frames: int = RELEASE_FRAMES
    input_sample_interval_ms: int = INPUT_SAMPLE_INTERVAL_MS
    output_sample_interval_ms: int = OUTPUT_SAMPLE_INTERVAL_MS
    max_frames: int = MAX_FRAMES


def strip_cpp_comments(text: str) -> str:
    """Remove simple C/C++ comments before number parsing."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def parse_uint8_sequence(text: str) -> List[int]:
    """Parse comma/newline separated integers and validate uint8 range."""
    cleaned = strip_cpp_comments(text)
    values = [int(match.group(0)) for match in INT_RE.finditer(cleaned)]
    bad = [value for value in values if value < 0 or value > 255]
    if bad:
        raise ValueError(f"Values must be in 0..255, got examples: {bad[:5]}")
    return values


def sort_stimulus_name(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def parse_pasted_stimuli(text: str) -> Dict[str, List[int]]:
    """Parse pasted raw text as Arduino arrays, or as one unnamed sequence."""
    source = text.strip()
    if not source:
        return {}

    sequences: Dict[str, List[int]] = {}
    for match in ARRAY_RE.finditer(source):
        name = match.group(1)
        sequences[name] = parse_uint8_sequence(match.group("body"))

    if sequences:
        return dict(sorted(sequences.items(), key=lambda item: sort_stimulus_name(item[0])))

    values = parse_uint8_sequence(source)
    if not values:
        return {}
    return {"STIMULUS_01": values}


def load_sequence(
    input_ino: Path | None,
    sequence_text: str | None,
    pasted_text: str,
) -> Dict[str, List[int]]:
    """Load one direct sequence, explicit .ino arrays, or pasted raw arrays."""
    if sequence_text:
        return {"STIMULUS_01": parse_uint8_sequence(sequence_text)}

    if input_ino is not None:
        source = input_ino.read_text(encoding="utf-8")
        sequences: Dict[str, List[int]] = {}
        for match in ARRAY_RE.finditer(source):
            name = match.group(1)
            sequences[name] = parse_uint8_sequence(match.group("body"))

        if not sequences:
            raise ValueError(f"No STIMULUS_XX arrays found in {input_ino}")

        return dict(sorted(sequences.items(), key=lambda item: sort_stimulus_name(item[0])))

    pasted_sequences = parse_pasted_stimuli(pasted_text)
    if pasted_sequences:
        return pasted_sequences

    raise ValueError(
        "Paste raw data into PASTED_STIMULI, or provide --sequence / --input-ino."
    )


def clamp_uint8(value: float) -> int:
    """Round and clamp a numeric value into uint8 range."""
    return max(0, min(255, int(value + 0.5)))


def perceptual_map(samples: Sequence[int], params: Params) -> List[int]:
    """
    Convert raw direct-amplitude uint8 values into a stronger ERM envelope.

    Values below threshold become 0. Kept values are gamma-mapped into
    [min_nonzero, max_drive], so weak-but-valid pulses remain perceivable.
    """
    mapped: List[int] = []
    drive_range = params.max_drive - params.min_nonzero

    for value in samples:
        if value < params.threshold:
            mapped.append(0)
            continue

        normalized = value / 255.0
        shaped = params.min_nonzero + drive_range * math.pow(normalized, params.gamma)
        mapped.append(max(params.min_nonzero, min(params.max_drive, clamp_uint8(shaped))))

    return mapped


def resample_interval(samples: Sequence[int], params: Params) -> List[int]:
    """
    Convert input-frame data to the output interval with peak-hold pooling.

    For the default 10 ms -> 20 ms case, each output frame becomes:
      max(input[0], input[1]), max(input[2], input[3]), ...

    Max pooling is intentional for haptics: it preserves short peaks that would
    become weaker if two frames were averaged.
    """
    if params.input_sample_interval_ms == params.output_sample_interval_ms:
        return list(samples)

    if params.input_sample_interval_ms <= 0 or params.output_sample_interval_ms <= 0:
        raise ValueError("Sample intervals must be positive.")

    total_duration_ms = len(samples) * params.input_sample_interval_ms
    output_count = math.ceil(total_duration_ms / params.output_sample_interval_ms)
    resampled: List[int] = []

    for out_index in range(output_count):
        window_start_ms = out_index * params.output_sample_interval_ms
        window_end_ms = window_start_ms + params.output_sample_interval_ms
        first_input = window_start_ms // params.input_sample_interval_ms
        last_input = math.ceil(window_end_ms / params.input_sample_interval_ms)
        window = samples[first_input:last_input]
        resampled.append(max(window) if window else 0)

    return resampled


def enforce_min_pulse(samples: Sequence[int], params: Params) -> List[int]:
    """
    Extend short non-zero pulses to the minimum frame length.

    Extension happens into nearby zero frames only, so existing neighboring
    pulse shapes are not overwritten.
    """
    out = list(samples)
    min_frames = params.min_pulse_frames
    if min_frames <= 1:
        return out

    index = 0
    while index < len(out):
        while index < len(out) and out[index] == 0:
            index += 1
        if index >= len(out):
            break

        start = index
        peak = out[index]
        while index < len(out) and out[index] > 0:
            peak = max(peak, out[index])
            index += 1
        end = index

        length = end - start
        if length >= min_frames:
            continue

        needed = min_frames - length

        after = end
        while needed > 0 and after < len(out) and out[after] == 0:
            out[after] = peak
            after += 1
            needed -= 1

        before = start
        while needed > 0 and before > 0 and out[before - 1] == 0:
            before -= 1
            out[before] = peak
            needed -= 1

        # Skip the frames we just extended forward, so one tiny pulse does not
        # get repeatedly extended during the same pass.
        index = max(index, after)

    return out


def smooth_attack_release(samples: Sequence[int], params: Params) -> List[int]:
    """Apply optional linear attack/release smoothing to each non-zero pulse."""
    out = list(samples)
    if params.attack_frames <= 0 and params.release_frames <= 0:
        return out

    index = 0
    while index < len(out):
        while index < len(out) and out[index] == 0:
            index += 1
        if index >= len(out):
            break

        start = index
        while index < len(out) and out[index] > 0:
            index += 1
        end = index
        length = end - start

        for offset in range(length):
            factor = 1.0

            if params.attack_frames > 0 and offset < params.attack_frames:
                attack_factor = (offset + 1) / (params.attack_frames + 1)
                factor = min(factor, attack_factor)

            remaining = length - offset - 1
            if params.release_frames > 0 and remaining < params.release_frames:
                release_factor = (remaining + 1) / (params.release_frames + 1)
                factor = min(factor, release_factor)

            if factor < 1.0:
                value = clamp_uint8(out[start + offset] * factor)
                if value > 0:
                    value = max(params.min_nonzero, value)
                out[start + offset] = value

    return out


def preprocess_sequence(samples: Sequence[int], params: Params) -> List[int]:
    """Run the complete ERM preprocessing pipeline on one sequence."""
    limited = list(samples[: params.max_frames]) if params.max_frames > 0 else list(samples)
    resampled = resample_interval(limited, params)
    mapped = perceptual_map(resampled, params)
    pulsed = enforce_min_pulse(mapped, params)
    return smooth_attack_release(pulsed, params)


def preprocess_all(
    sequences: Mapping[str, Sequence[int]],
    params: Params,
) -> Dict[str, List[int]]:
    """Preprocess all loaded arrays."""
    return {name: preprocess_sequence(values, params) for name, values in sequences.items()}


def format_arduino_array(name: str, samples: Sequence[int], values_per_line: int = 16) -> str:
    """Format one const uint8_t array for copy/paste into Arduino code."""
    lines = [f"const uint8_t {name}[] = {{"]
    for start in range(0, len(samples), values_per_line):
        chunk = samples[start : start + values_per_line]
        suffix = "," if start + values_per_line < len(samples) else ""
        lines.append("  " + ",".join(str(value) for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def export_to_arduino_header(
    path: Path,
    sequences: Mapping[str, Sequence[int]],
    params: Params,
) -> None:
    """Write a paste-ready Arduino fragment containing processed arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "// ERM-ready arrays generated by scripts/erm_preprocess_waveforms.py",
        "// Paste these STIMULUS_* arrays over the raw arrays in esp32_haptic_panel.ino.",
        f"// input_sample_interval_ms={params.input_sample_interval_ms}, output_sample_interval_ms={params.output_sample_interval_ms}, max_input_frames={params.max_frames}",
        "// IMPORTANT: set DEFAULT_SAMPLE_INTERVAL_MS in esp32_haptic_panel.ino to the output interval above.",
        f"// threshold={params.threshold}, gamma={params.gamma}, min_nonzero={params.min_nonzero}, max_drive={params.max_drive}",
        f"// min_pulse_frames={params.min_pulse_frames}, attack_frames={params.attack_frames}, release_frames={params.release_frames}",
        "",
    ]
    for name, samples in sequences.items():
        parts.append(format_arduino_array(name, samples))
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def export_to_json(
    path: Path,
    raw_sequences: Mapping[str, Sequence[int]],
    processed_sequences: Mapping[str, Sequence[int]],
    params: Params,
) -> None:
    """Write processed data and parameters for inspection/reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_sample_interval_ms": params.input_sample_interval_ms,
        "output_sample_interval_ms": params.output_sample_interval_ms,
        "sample_interval_ms": params.output_sample_interval_ms,
        "params": asdict(params),
        "raw_sequences": {name: list(values) for name, values in raw_sequences.items()},
        "processed_sequences": {name: list(values) for name, values in processed_sequences.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_sequences(
    plot_dir: Path,
    raw_sequences: Mapping[str, Sequence[int]],
    processed_sequences: Mapping[str, Sequence[int]],
    params: Params,
) -> None:
    """Optionally save one PNG comparison per sequence. Requires matplotlib."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in raw_sequences.items():
        processed = processed_sequences[name]
        raw_t = [
            index * params.input_sample_interval_ms / 1000.0 for index in range(len(raw))
        ]
        processed_t = [
            index * params.output_sample_interval_ms / 1000.0
            for index in range(len(processed))
        ]

        plt.figure(figsize=(12, 4))
        plt.plot(raw_t, raw, label="raw", linewidth=1.5)
        plt.plot(processed_t, processed, label="ERM-ready", linewidth=1.5)
        plt.title(name)
        plt.xlabel("Time (s)")
        plt.ylabel("uint8 drive")
        plt.ylim(-5, 260)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png", dpi=150)
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw Arduino uint8 haptic arrays into ERM-ready arrays."
    )
    parser.add_argument(
        "--input-ino",
        type=Path,
        help="Optional Arduino .ino file to read. Default is the PASTED_STIMULI block.",
    )
    parser.add_argument("--sequence", help="Optional direct comma-separated uint8 sequence.")
    parser.add_argument(
        "--output-arduino",
        type=Path,
        default=Path("outputs/erm_ready_stimuli.h"),
        help="Paste-ready Arduino array output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/erm_ready_stimuli.json"),
        help="JSON output path for inspection.",
    )
    parser.add_argument("--plot-dir", type=Path, help="Optional directory for PNG plots.")
    parser.add_argument("--threshold", type=int, default=THRESHOLD)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--min-nonzero", type=int, default=MIN_NONZERO)
    parser.add_argument("--max-drive", type=int, default=MAX_DRIVE)
    parser.add_argument("--min-pulse-frames", type=int, default=MIN_PULSE_FRAMES)
    parser.add_argument("--attack-frames", type=int, default=ATTACK_FRAMES)
    parser.add_argument("--release-frames", type=int, default=RELEASE_FRAMES)
    parser.add_argument(
        "--input-sample-interval-ms",
        type=int,
        default=INPUT_SAMPLE_INTERVAL_MS,
        help="Interval of pasted/raw input samples.",
    )
    parser.add_argument(
        "--output-sample-interval-ms",
        "--sample-interval-ms",
        dest="output_sample_interval_ms",
        type=int,
        default=OUTPUT_SAMPLE_INTERVAL_MS,
        help="Interval of generated ERM-ready samples.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help="Keep at most this many frames. Use 0 for no limit.",
    )
    return parser


def validate_params(params: Params) -> None:
    if not 0 <= params.threshold <= 255:
        raise ValueError("--threshold must be in 0..255")
    if params.gamma <= 0:
        raise ValueError("--gamma must be > 0")
    if not 0 <= params.min_nonzero <= 255:
        raise ValueError("--min-nonzero must be in 0..255")
    if not 0 <= params.max_drive <= 255:
        raise ValueError("--max-drive must be in 0..255")
    if params.min_nonzero > params.max_drive:
        raise ValueError("--min-nonzero must be <= --max-drive")
    if params.min_pulse_frames < 1:
        raise ValueError("--min-pulse-frames must be >= 1")
    if params.attack_frames < 0 or params.release_frames < 0:
        raise ValueError("--attack-frames and --release-frames must be >= 0")
    if params.input_sample_interval_ms <= 0:
        raise ValueError("--input-sample-interval-ms must be > 0")
    if params.output_sample_interval_ms <= 0:
        raise ValueError("--output-sample-interval-ms must be > 0")


def main() -> int:
    args = build_arg_parser().parse_args()
    params = Params(
        threshold=args.threshold,
        gamma=args.gamma,
        min_nonzero=args.min_nonzero,
        max_drive=args.max_drive,
        min_pulse_frames=args.min_pulse_frames,
        attack_frames=args.attack_frames,
        release_frames=args.release_frames,
        input_sample_interval_ms=args.input_sample_interval_ms,
        output_sample_interval_ms=args.output_sample_interval_ms,
        max_frames=args.max_frames,
    )
    validate_params(params)

    raw_sequences = load_sequence(args.input_ino, args.sequence, PASTED_STIMULI)
    processed_sequences = preprocess_all(raw_sequences, params)

    export_to_arduino_header(args.output_arduino, processed_sequences, params)
    export_to_json(args.output_json, raw_sequences, processed_sequences, params)

    if args.plot_dir:
        plot_sequences(args.plot_dir, raw_sequences, processed_sequences, params)

    print(f"Loaded sequences: {len(raw_sequences)}")
    print(f"Wrote Arduino arrays: {args.output_arduino}")
    print(f"Wrote JSON: {args.output_json}")
    if args.plot_dir:
        print(f"Wrote plots: {args.plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
