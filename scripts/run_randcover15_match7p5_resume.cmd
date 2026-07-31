@echo off
set ROOT=C:\pycharm\pointdiff_new
set PYTHON=C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe
set CONFIG=%ROOT%\config\train_randcover15_match7p5_from_0073.yaml
set STATUS=%ROOT%\logs\randcover15_match7p5_task_status.txt
set STDOUT=%ROOT%\logs\randcover15_match7p5_task_stdout.log
set STDERR=%ROOT%\logs\randcover15_match7p5_task_stderr.log

cd /d "%ROOT%"
echo state=running > "%STATUS%"
echo started_at=%DATE% %TIME% >> "%STATUS%"
echo config=%CONFIG% >> "%STATUS%"
echo stdout=%STDOUT% >> "%STATUS%"
echo stderr=%STDERR% >> "%STATUS%"

"%PYTHON%" "%ROOT%\main.py" --config "%CONFIG%" 1> "%STDOUT%" 2> "%STDERR%"
set EXITCODE=%ERRORLEVEL%

echo state=exited > "%STATUS%"
echo ended_at=%DATE% %TIME% >> "%STATUS%"
echo exit_code=%EXITCODE% >> "%STATUS%"
echo config=%CONFIG% >> "%STATUS%"
echo stdout=%STDOUT% >> "%STATUS%"
echo stderr=%STDERR% >> "%STATUS%"
exit /b %EXITCODE%
