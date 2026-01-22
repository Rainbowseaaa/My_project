# -*- coding: utf-8 -*-

#------------------------- \cond COPYRIGHT --------------------------#
#                                                                    #
# Copyright (C) 2025 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#----------------------------- \endcond -----------------------------#


# This example contains multiple examples of preselection strings for usage in SLM window initialization.
# Please see the PDF manual for documentation on the available options and maybe additional example strings.
# The main use case of preselection strings are:
# - Preselection of a specific SLM device in multi-SLM setups.
# - Change special device or rendering settings, like enabling or disabling 10-bit addressing modes,
# which are only supported by some SLM devices, like ERIS.
#
# Please note that to make ERIS 10-bit modes work, the selected mode also needs to be set up in
# ERIS Configuration Manager, and the device must be operated in the correct 60Hz or 120 Hz mode.
# There is also a 10-bit mode for hardware-changed PLUTO-1 10-bit devices, which does not make sense to use this with ERIS.
#
# Addressing 10-bit data on 10-bit devices requires to upload data in float format, which is done by the built-in
# load/show functions, if applicable.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# Print SDK version info of the connected SDK library file:
HEDS.SDK.PrintVersion()

# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,1)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window with default scale (1:1) for the selected SLM.
# Add preselection strings to enable 10-bit output (please uncomment the line you want to try):

# Must be used with ERIS 60Hz when input color channel in ERIS CM is configured to 8-bit on red and lower 2 bit on green.
slm = HEDS.SLM.Init("-slm name:eris -nogui -osm MonoR10bit_R8G2LSB")

# Must be used with ERIS 60Hz when input color channel in ERIS CM is configured to 8-bit on green and lower 2 bit on blue.
#  auto slm = HEDS::SLM::Init("-slm name:eris -nogui -osm MonoR10bit_G8B2LSB");

# Must be used with PLUTO-1 10-bit hardware to make sense.
#  auto slm = HEDS::SLM::Init("-slm \"name:holoeye pluto slm\" -nogui -osm MonoR10bit_G8B2MSB");

# Must be used with ERIS 120Hz. This switches off 10-bit addressing. It is recommended to leave the ERIS 120Hz in 10-bit mode,
# but for testing this option can be used to switch the default 10-bit addressing of 120Hz mode back to 8-bit addressing.
#  auto slm = HEDS::SLM::Init("-slm name:eris -nogui -osm ERIS120Hz_8bit");

# Evaluate the error code of the selected HEDS::SLM::Init() call:
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Set the SLM preview into screen capture mode here to visualize the different color addressing modes for 10-bit devices:
err = slm.preview().setCaptureMode(HEDSSLMPM_CaptureScreen)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Configure the lens phase function:
innerRadiusPx = 120

# Show built-in Fresnel lens phase function:
# Built-in functions are capable of 10-bit values (by using float data format).
err = slm.showPhaseFunctionAxicon(innerRadiusPx)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

