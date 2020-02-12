"""
Definitions of constant values that will be used through all the project

"""
# use Box Codes for the delimitation to avoid any encoding collition
SEPARATOR = '█'  # to use as separator instead of CSV or TSV
CHAR_MASK = '▒'  # To use as a mask for a character
# some interesting characters for use:
# ▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅
# ◆◇◈◉◊○◌◍◎●◐◑◒◓
# Begining Control Codes used for special purposes
RESERVED_CODE_SPACE = 33  # the n# 32 is the space
# \t, \r, \n and ' ' (space) are in this space, NOT TO OVERWRITE
NUL = ('◌', 0x00, '◁NUL▷')  # NUL control code -> for Padding for example
SOH = ('◀', 0x01, '◁SOH▷')  # SOH control code (Start of Heading) -> example: to indicate a task description or tgt lang
STX = ('◂', 0x02, '◁STX▷')  # STX control code (Start of Text) -> start of text
ETX = ('▸', 0x03, '◁ETX▷')  # ETX control code (End of Text) -> end of text
EOT = ('▶', 0x04, '◁EOT▷')  # EOT control code (End of Transmission) -> end of document
UNK = ('◍', 0x15, '◁UNK▷')  # NAK control code (Negative Acknowledge) -> Unknown value
# SUB = ('◁SUB▷', 0x1A)  # SUB control code (Substitute) -> Garbled or Invalid Characters
MSK = ('▒', 0x1A, '◁MSK▷')  # Use this instead as mask for a single character
UNASSIGNED = '◁???▷'

SPECIAL_CODES = (NUL, UNK, SOH, STX, ETX, EOT, MSK)
# SPECIAL_CODES = (NUL, UNK, SOH, STX, ETX, EOT, SUB, MSK)
