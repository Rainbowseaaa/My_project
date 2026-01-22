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
# I.e. you pass an image to show in the far field and the phase function is calculated internally using
# the Gerchberg-Saxton algorithm (Iterative Fourier Transform Algorithm, IFTA).


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

# Configure the spot matrix signal:
# 7 spots in x-direction
nx = 7
# 7 spots in y-direction.
ny = 7
# will use either embedding or downscaling to adjust the size in far field image. Must be up to 1.0 to make sense.
scale = 0.25

# Generate a slot matrix in far field:
imgdata = HEDS.SLMDataField(nx, ny, HEDSDTFMT_INT_U8)

# Fill image with white pixels:
for j in range(0, ny):
    for i in range(0, nx):
        err = imgdata.setPixel(i, j, 255)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

err, datahandle = slm.loadCGHFromImageData(imgdata.data(), HEDSLDF_Default | HEDSSHF_PresentTiledCentered, scale, int(slm.height_px()/2))
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

err = datahandle.show()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

