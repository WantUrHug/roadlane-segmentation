@echo off
title using labelme...
::if "%1" == "/a" echo first param is /a
::dir
echo we need to do %1 json files.
for /l %%i in (1,1,%1) do (
if %%i LSS 10 (
labelme_json_to_dataset 0000%%i.json
) else (if %%i LSS 100 (labelme_json_to_dataset 000%%i.json))
)