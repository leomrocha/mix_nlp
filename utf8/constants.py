"""
Definitions of constant values that will be used through all the project

"""
# use Box Codes for the delimitation to avoid any encoding collition
NUL = ('◁NUL▷', 0x00)  # NUL control code -> for Padding for example
SOH = ('◁SOH▷', 0x01)  # SOH control code (Start of Heading) -> for example to indicate a task description
STX = ('◁STX▷', 0x02)  # STX control code (Start of Text) -> start of text
ETX = ('◁ETX▷', 0x03)  # ETX control code (End of Text) -> end of text
EOT = ('◁EOT▷', 0x04)  # EOT control code (End of Transmission) -> end of document
UNK = ('◁UNK▷', 0x15)  # NAK control code (Negative Acknowledge) -> Unknown value
UNASSIGNED = '◁???▷'

SPECIAL_CODES = (NUL, UNK, SOH, STX, ETX, EOT)
