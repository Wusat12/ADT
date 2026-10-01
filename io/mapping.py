# 3-class drum mapping for the thesis:
# 0 = bass drum
# 1 = snare drum
# 2 = hi-hat


ENST_MAPPING = {
    # Bass drum
    "bd": 0,

    # Snare drum
    "sd": 1,
    "sd-": 1,
    "sweep": 1,
    "rs": 1,
    "sticks": 1,
    "cs": 1,

    # Toms - excluded
    "mt": None,
    "mtr": None,
    "lmt": None,
    "lt": None,
    "ltr": None,
    "lft": None,

    # Hi-hat
    "chh": 2,
    "ohh": 2,

    # Other cymbals/percussion - excluded
    "cr": None,
    "spl": None,
    "ch": None,
    "rc": None,
    "c": None,
    "cb": None,
}


MDB_MAPPING = {
    # Bass drum
    "KD": 0,

    # Snare drum
    "SD": 1,
    "SDB": 1,
    "SDD": 1,
    "SDF": 1,
    "SDG": 1,
    "SDNS": 1,
    "SST": 1,

    # Toms - excluded
    "HIT": None,
    "MHT": None,
    "HFT": None,
    "LFT": None,

    # Hi-hat
    "CHH": 2,
    "OHH": 2,
    "PHH": 2,

    # Tambourine - excluded
    "TMB": None,

    # Cymbals - excluded
    "RDC": None,
    "RDB": None,
    "CRC": None,
    "CHC": None,
    "SPC": None,
}


# Kept for compatibility with the existing repository.
# These mappings are not used by the ENST+MDB conversion pipeline.
ROLAND_MIDI_MAPPING = {
    35: 0,
    36: 0,

    37: 1,
    38: 1,
    39: 1,
    40: 1,

    41: None,
    43: None,
    45: None,
    47: None,
    48: None,
    50: None,
    58: None,

    22: 2,
    26: 2,
    42: 2,
    44: 2,
    46: 2,

    54: None,

    49: None,
    51: None,
    52: None,
    53: None,
    55: None,
    56: None,
    57: None,
    59: None,

    27: None,
    28: None,
    29: None,
    30: None,
    31: None,
    32: None,
    33: None,
    34: None,
    60: None,
    61: None,
    62: None,
    63: None,
    64: None,
    65: None,
    66: None,
    67: None,
    68: None,
    69: None,
    70: None,
    71: None,
    72: None,
    73: None,
    74: None,
    75: None,
    76: None,
    77: None,
    78: None,
    79: None,
    80: None,
    81: None,
    82: None,
    83: None,
    84: None,
    85: None,
    86: None,
    87: None,
}


MIDI_MAPPING = {
    35: 0,
    36: 0,

    37: 1,
    38: 1,
    39: 1,
    40: 1,

    41: None,
    43: None,
    45: None,
    47: None,
    48: None,
    50: None,

    42: 2,
    44: 2,
    46: 2,

    54: None,

    49: None,
    51: None,
    52: None,
    53: None,
    55: None,
    56: None,
    57: None,
    59: None,

    27: None,
    28: None,
    29: None,
    30: None,
    31: None,
    32: None,
    33: None,
    34: None,
    58: None,
    60: None,
    61: None,
    62: None,
    63: None,
    64: None,
    65: None,
    66: None,
    67: None,
    68: None,
    69: None,
    70: None,
    71: None,
    72: None,
    73: None,
    74: None,
    75: None,
    76: None,
    77: None,
    78: None,
    79: None,
    80: None,
    81: None,
    82: None,
    83: None,
    84: None,
    85: None,
    86: None,
    87: None,
}