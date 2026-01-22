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


# This example demonstrates calculating phase data out of image data meant to be shown in far field.
# I.e. you pass an image file to show in the far field and the phase function is calculated internally using
# the Gerchberg-Saxton algorithm (Iterative Fourier Transform Algorithm, IFTA).
#
# For illumination, a monochromatic laser is used. The monochromatic image (albert.png) then serves as the basis
# for calculating the computer-generated hologram (CGH), which is displayed using a spatial light modulator (SLM).
# Subsequently, an RGB file is used, from which a corresponding CGH is also computed and displayed on the SLM.
# The resulting output remains monochromatic as well.

# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# Print SDK version info of the connected SDK library file:
HEDS.SDK.PrintVersion()

# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,1)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window with default scale (1:1) for the selected SLM:
slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Use an image file as a signal and calculate the CGH to be shown on the display:
filename = "data/albert.png"
err = slm.showCGHFromImageFile(filename)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print(" --> slm.showCGHFromImageFile(" + filename + ")")
if not slm.wait(2500) == HEDSERR_NoError:
    exit(1)

# Use an rgb image file as a signal and calculate the CGH to be shown on the display:
filename = "data/RGBCMY01_640x480.png"
err = slm.showCGHFromImageFile(filename)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
print(" --> slm.showCGHFromImageFile(" + filename + ")")
if not slm.wait(2500) == HEDSERR_NoError:
    exit(1)


# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

