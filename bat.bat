@echo off

for /f "tokens=*" %%a in (C:\xampp\htdocs\TVAAKU-master\new.txt) do set varText=%%a

set varText=%varText%
python C:\xampp\htdocs\TVAAKU-master\knn_predict.py "%varText%" %*
REM ECHO %varText%
break>c:\'C:\xampp\htdocs\TVAAKU-master'\demo.txt
pause
