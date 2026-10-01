NUM_FEATURES = 168  # 84 Mel bands + 84 positive deltas
ENCODER_OUT = 64 * (NUM_FEATURES // 9)  # 2 conv blocks: 64 channels, (1,3) pooling twice
NUM_CLASSES = 3  # bass drum, snare drum, hi-hat
