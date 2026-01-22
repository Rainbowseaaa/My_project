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


# Uses the built-in blank screen function to show a given grayscale value on the full SLM.
# Then we use the beam manipulation provided through the data handles to apply a phase
# overlay (tip/tilt/lens) calculated from physical units.
#
# This example applies beam manipulation after the data handle is already visible.
# Of course, beam manipulation values can be applied before showing the data handle for
# the first time, and then beam manipulation is visible right away with the data.


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

# Configure beam manipulation in physical units:
wavelength_nm = 633.0   # wavelength of incident laser light

steering_angle_x_deg = 0.2
steering_angle_y_deg = -0.3
focal_length_mm = 200.0

# Apply wavelength to SLM to be able to convert physical units into generalized units for beam manipulation.
# This also enables phase modulation for the SLM, which is necessary to apply beam manipulation.
# In amplitude modulation, beam manipulation values would be ignored:
err = slm.setWavelength(wavelength_nm)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Upload a blank screen into the SLM. This will create an SLMDataHandle (dh):
err, dh = slm.loadBlankScreen(grayValue)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Make the data visible on the SLM screen without beam manipulation overlay:
err = dh.show(HEDSSHF_PresentAutomatic)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Calculate proper generalized beam manipulation parameters to pass to the data handle.
# The handle parameters are related to SLM properties and are scaled to produce
# meaningful results in the range between -1.0f and 1.0f. The class BeamManipulation
# allows us to use the heds_utils_beam_xxx functions to convert between the physical
# observables (deviation angles and focal length of the lens) and these generalized
# parameters:
bm = HEDS.BeamManipulation(0.0, 0.0, 0.0, 0, 0.0, 0.0, None, dh)
bm.setBeamSteerXDegree(steering_angle_x_deg)
bm.setBeamSteerYDegree(steering_angle_y_deg)
bm.setBeamLensFocalLengthMM(focal_length_mm)

# Print generalized parameters and back-converted observables for validation:
print("bm.getBeamSteerX() = %f" %bm.getBeamSteerX() + " ==> steering angle x = %f deg" %bm.getBeamSteerXDegree())
print("bm.getBeamSteerY() = %f" %bm.getBeamSteerY() + " ==> steering angle y = %f deg" %bm.getBeamSteerYDegree())
print("bm.getBeamLens()   = %f" %bm.getBeamLens()   + " ==> f = %f mm" %bm.getBeamLensFocalLengthMM())

# After calculating the generalized beam manipulation parameters, we can apply them to the data handle:
err = dh.setBeamManipulation(bm)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Now the new beam manipulation can be made active on the already visible data (handle),
# and the result will be visible on screen within the next video frame:
err = dh.apply()
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)


# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

