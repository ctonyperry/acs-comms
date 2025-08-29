REM Development tasks for ACS Bridge (Windows PowerShell)

@echo off
if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="install-dev" goto install-dev
if "%1"=="install-all" goto install-all
if "%1"=="fmt" goto fmt
if "%1"=="lint" goto lint
if "%1"=="type" goto type
if "%1"=="test" goto test
if "%1"=="run" goto run
if "%1"=="run-uvicorn" goto run-uvicorn
if "%1"=="clean" goto clean
if "%1"=="setup-hooks" goto setup-hooks
if "%1"=="all-checks" goto all-checks
goto help

:help
echo Available commands:
echo   install      - Install production dependencies
echo   install-dev  - Install development dependencies  
echo   install-all  - Install all dependencies (prod + dev + optional)
echo   fmt          - Format code with black
echo   lint         - Lint code with ruff
echo   type         - Type check with mypy
echo   test         - Run tests
echo   run          - Run the application locally
echo   run-uvicorn  - Run with uvicorn directly
echo   clean        - Clean up build artifacts and cache
echo   setup-hooks  - Setup pre-commit hooks
echo   all-checks   - Run all checks (format, lint, type, test)
goto end

:install
pip install -r requirements.txt
goto end

:install-dev
pip install -e ".[dev]"
goto end

:install-all
pip install -e ".[dev,stt,tts]"
goto end

:fmt
black src/ tests/ run.py
goto end

:lint
ruff check src/ tests/ run.py
goto end

:type
mypy src/
goto end

:test
pytest tests/
goto end

:run
python run.py
goto end

:run-uvicorn
uvicorn src.acs_bridge.main:app --reload --host 0.0.0.0 --port 8080
goto end

:clean
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .mypy_cache rmdir /s /q .mypy_cache
if exist .ruff_cache rmdir /s /q .ruff_cache
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul
goto end

:setup-hooks
pre-commit install
goto end

:all-checks
call %0 fmt
call %0 lint
call %0 type
call %0 test
goto end

:end