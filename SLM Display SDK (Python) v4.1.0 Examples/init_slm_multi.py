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


# This example opens two differnt SLM devices in parallel. For each SLM a separate preview window is opened.
# Each SLM shows a vortex using the built-in function. Each vortex is shown with a different order to make them distinguishable.


# Import HOLOEYE SLM Display SDK:
import HEDS
from hedslib.heds_types import *

# Print SDK version info of the connected SDK library file:
HEDS.SDK.PrintVersion()

# Check and Init for proper SDK Version:
err = HEDS.SDK.Init(4,1)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Retrieve the first SLM found by EDID device detection in the system, and open the SLM preview window for the selected SLM.
# If the preselected SLM with index 0 is found, it is used right away and no GUI selection will open here.
# Otherwise, an error code is generated due to the additional option -nogui. Without that option, EDID Device Detection GUI would open in the error case.
# Preselect string examples:
#    "index:0"  // select first SLM available in the system.
#    "name:pluto;serial:0001"  // select a PLUTO SLM with the serial number 0001.
#    "name:luna"  // select a LUNA SLM.
#    "serial:2220-0011"  // select a LUNA/2220 SLM just by passing the serial number.
slmFirst = HEDS.SLM.Init("-slm index:0 -nogui")
assert slmFirst.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmFirst.errorCode())

# Call device detection again, and retrieve the second SLM found in the system, then open the SLM preview window for the selected SLM.
# If there is a second SLM found in the system, the second SLM is used right away and no GUI selection will open here.
# Otherwise, an error code is generated due to the additional option -nogui. Without that option, EDID Device Detection GUI would open in the error case.
slmSecond = HEDS.SLM.Init("-slm index:1 -nogui")
assert slmSecond.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmSecond.errorCode())

# Move both SLM preview windows so that they are not overlapping, and are placed on a secondary/primary monitor (and not an SLM screen):

# Print out found information about the monitors in the system on command line:
HEDS.SDK.PrintMonitorInfos();

# Set up and move the first SLM preview window:
err = slmFirst.preview().autoplaceLayoutOnSecondaryMonitor( 2, 1, 0, 0)  # col,row,col,row
assert slmFirst.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmFirst.errorCode())

err = slmFirst.preview().setSettings(HEDSSLMPF_None, 0.0)
assert slmFirst.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmFirst.errorCode())

# Setup and move the second SLM preview window:
err = slmSecond.preview().autoplaceLayoutOnSecondaryMonitor( 2, 1, 1, 0)  # col,row,col,row
assert slmSecond.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmSecond.errorCode())

err = slmSecond.preview().setSettings(HEDSSLMPF_None, 0.0)
assert slmSecond.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmSecond.errorCode())

# A list of data handles collecting all preloaded data to later make them visible on the SLM as simultaneous as possible:
handles = []

# Prepare some data for the first SLM using the built-in vortex phase function with an order of 1:
err, slmDataHandle = slmFirst.loadVortex(1)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
handles.append(slmDataHandle)

# Prepare some data for the second SLM using the built-in vortex phase function with an order of 2:
err, slmDataHandle = slmSecond.loadVortex(2)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
handles.append(slmDataHandle)

# Display both data handles at once:
err = HEDS.SDK.ShowDataHandles(handles)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Wait until each opened SLM window was closed manually by using the tray icon GUI:
HEDS.SDK.WaitAllClosed()

