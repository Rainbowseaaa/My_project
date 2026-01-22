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


# This example demonstrates how to apply a wavefront compensation from a *.h5 file provided by HOLOEYE together with the device.
# For demonstration purpose there is a demo-file included in this example, which will not improve the wavefront shape for any
# device, but is clearly visible.



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

# Configure the blank screen where overlay is applied to:
grayValue = 128

# Retrieve an SLM window:
slmWindow = slm.window()

# Increase SLM preview size, and if available, place it on a secondary monitor:
err = slmWindow.preview().autoplaceLayoutOnSecondaryMonitor()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Enable wavefrontcompensation visualization in SLM preview window and stay with SLM preview scale "Fit":
err = slmWindow.preview().setSettings(HEDSSLMPF_ShowWavefrontCompensation, 0.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Give each SLM an incident laser wavelength, so that the wavefront compensation data loaded from file can be converted into phase values:
err = slm.setWavelength(532.0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Show a blank screen on the entire SLM:
err = slm.showBlankScreen(grayValue)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Load a demo wavefront compensation file (*.h5). This must be changed to the file provided for your specifically used SLM,
# and in case of multiple SLMs, it needs to be the one for the correct SLM, of course:
wavefrontCompensationFilename = "data/wfcdemo_holoeye_logo.h5"
err = slmWindow.loadWavefrontCompensationFile(wavefrontCompensationFilename)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until opened SLM window was closed manually by using the tray icon GUI:
print("Wavefront compensation is now applied on SLM display device.")

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

