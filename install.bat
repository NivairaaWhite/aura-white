@echo off
rem Aura White - one-click Windows installer. Creates a private .venv so nothing else on the PC is touched.
rem   install.bat            -> CPU (or whatever PyTorch pip picks by default)
rem   install.bat cu126      -> NVIDIA GPU build of PyTorch (also: cu124, cu128 ...)
rem   add the word  studio  as the last argument to also install the texturing extras, e.g.  install.bat cu126 studio
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .venv || (echo Could not create a virtual environment. Install Python 3.9+ from python.org and retry. & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
set STUDIO=0
if /i "%~1"=="studio" set STUDIO=1
if /i "%~2"=="studio" set STUDIO=1
if not "%~1"=="" if /i not "%~1"=="studio" (
  echo Installing PyTorch for %~1 ...
  pip install torch --index-url https://download.pytorch.org/whl/%~1 || exit /b 1
)
pip install -e ".[recommended]" || pip install -e .
if "%STUDIO%"=="1" (
  echo Installing the texturing extras ...
  pip install -r requirements-studio.txt || echo Some optional extras failed - the studio uses built-in fallbacks for those.
)
echo.
echo ---- checking the install ----
python -m aura_white doctor
echo.
echo Next:  run.bat serve        (web app with 3D viewer)
echo        run.bat studio art.png   (sharp textured asset)
echo        run.bat photo.png    (command line)
endlocal
