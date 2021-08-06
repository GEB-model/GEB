@echo off
setlocal EnableDelayedExpansion

FOR /L %%I IN (1, 1, 26) DO (
     set "formattedValue=000000%%I"
     echo !formattedValue:~-2!
     flt_to_asc.exe crop_!formattedValue:~-2!_irrigated_12.flt 4320 2160 12
)

FOR /L %%I IN (1, 1, 26) DO (
     set "formattedValue=000000%%I"
     echo !formattedValue:~-2!
     flt_to_asc.exe crop_!formattedValue:~-2!_rainfed_12.flt 4320 2160 12
)

pause