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
# The specific wavelengths emitted by the FISBA laser source, as well as the intrinsic chromatic aberration
# characteristics inherent to the optical system, are predefined. Subsequently, a monochromatic image is numerically
# computed in the form of a computer-generated hologram (CGH) and displayed via a spatial light modulator (SLM).
# The resultant output is an RGB-based CGH.
# In a final step, an RGB image file serves as input data, from which the corresponding CGH is calculated and subsequently rendered on the SLM.

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

# Default values for FISBA laser
slm.setWavelengthCFS(638.0, 520.0, 450.0)
# Allows to compensate for transverse (lateral) chromatic abberations within the optical setup when using color field sequential mode on the SLM screen:
slm.setChromaticAbberationFactorsCFS(1.0, 1.0, 1.0)

# Use an monochromatic image file as a signal and calculate the CGH to be shown on the display:
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

