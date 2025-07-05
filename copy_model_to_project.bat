@echo off
echo 🚀 COPY YOUR FINE-TUNED MODEL TO ANY PROJECT
echo ================================================

if "%~1"=="" (
    set /p TARGET_PATH="Enter target project path: "
) else (
    set TARGET_PATH=%~1
)

if "%TARGET_PATH%"=="" (
    echo ❌ No target path provided
    pause
    exit /b 1
)

echo.
echo 📂 Target project: %TARGET_PATH%
echo.

REM Create target directory
if not exist "%TARGET_PATH%" (
    mkdir "%TARGET_PATH%"
    echo ✅ Created project directory
)

REM Create models directory
if not exist "%TARGET_PATH%\models" (
    mkdir "%TARGET_PATH%\models"
    echo ✅ Created models directory
)

REM Copy the model
echo 📦 Copying fine-tuned model...
xcopy "E:\Intel Fest\Fine Tune\checkpoints\codet5p-finetuned" "%TARGET_PATH%\models\codet5p-finetuned\" /E /I /Y
if %ERRORLEVEL% EQU 0 (
    echo ✅ Model copied successfully
) else (
    echo ❌ Error copying model
    pause
    exit /b 1
)

REM Copy the portable assistant
echo 📄 Copying code assistant...
copy "E:\Intel Fest\Fine Tune\portable_code_assistant.py" "%TARGET_PATH%\code_assistant.py"
if %ERRORLEVEL% EQU 0 (
    echo ✅ Code assistant copied
) else (
    echo ⚠️  Could not copy code assistant
)

REM Create requirements.txt
echo 📋 Creating requirements.txt...
echo torch^>=2.0.0> "%TARGET_PATH%\requirements.txt"
echo transformers^>=4.30.0>> "%TARGET_PATH%\requirements.txt"
echo accelerate^>=0.20.0>> "%TARGET_PATH%\requirements.txt"
echo flask^>=2.0.0>> "%TARGET_PATH%\requirements.txt"
echo ✅ Requirements.txt created

REM Create simple usage example
echo 💻 Creating example usage...
echo """>> "%TARGET_PATH%\use_model.py"
echo Simple example of using your fine-tuned model>> "%TARGET_PATH%\use_model.py"
echo """>> "%TARGET_PATH%\use_model.py"
echo.>> "%TARGET_PATH%\use_model.py"
echo from code_assistant import PortableCodeAssistant>> "%TARGET_PATH%\use_model.py"
echo.>> "%TARGET_PATH%\use_model.py"
echo # Initialize your model>> "%TARGET_PATH%\use_model.py"
echo assistant = PortableCodeAssistant()>> "%TARGET_PATH%\use_model.py"
echo.>> "%TARGET_PATH%\use_model.py"
echo # Complete some code>> "%TARGET_PATH%\use_model.py"
echo code = "def fibonacci(n):">> "%TARGET_PATH%\use_model.py"
echo completion = assistant.complete_code(code)>> "%TARGET_PATH%\use_model.py"
echo print(f"Input: {code}")>> "%TARGET_PATH%\use_model.py"
echo print(f"Output: {completion}")>> "%TARGET_PATH%\use_model.py"
echo ✅ Example file created

echo.
echo 🎉 SETUP COMPLETE!
echo ==================
echo.
echo 📁 Your model is now in: %TARGET_PATH%\models\codet5p-finetuned\
echo 🐍 Code assistant: %TARGET_PATH%\code_assistant.py
echo 📋 Requirements: %TARGET_PATH%\requirements.txt
echo 💻 Example: %TARGET_PATH%\use_model.py
echo.
echo 📋 Next steps:
echo 1. cd "%TARGET_PATH%"
echo 2. pip install -r requirements.txt
echo 3. python use_model.py
echo.
echo 🚀 Your fine-tuned model is ready to use in the new project!
echo.
pause
