@echo off
setlocal
cd /d "%~dp0"
set OUT=_git_review_scope.txt
> "%OUT%" echo ===== 1. git status -sb =====
git status -sb >> "%OUT%" 2>&1
>> "%OUT%" echo.
>> "%OUT%" echo ===== 2. git log --oneline -10 =====
git log --oneline -10 >> "%OUT%" 2>&1
>> "%OUT%" echo.
>> "%OUT%" echo ===== 3. git diff HEAD =====
git diff --name-only HEAD > _names_head.txt 2>&1
for /f "usebackq delims=" %%F in ("_names_head.txt") do (
  if not "%%F"=="" git diff HEAD -- "%%F" >> "%OUT%" 2>&1
)
>> "%OUT%" echo.
>> "%OUT%" echo ===== 4. git diff --cached =====
git diff --cached --name-only > _names_cached.txt 2>&1
for /f "usebackq delims=" %%F in ("_names_cached.txt") do (
  if not "%%F"=="" git diff --cached -- "%%F" >> "%OUT%" 2>&1
)
>> "%OUT%" echo.
>> "%OUT%" echo ===== 5. git diff origin/main...HEAD =====
git diff --name-only origin/main...HEAD > _names_range.txt 2>&1
for /f "usebackq delims=" %%F in ("_names_range.txt") do (
  if not "%%F"=="" git diff origin/main...HEAD -- "%%F" >> "%OUT%" 2>&1
)
>> "%OUT%" echo.
findstr /B /C:"diff --git" "%OUT%" >nul
if errorlevel 1 (
  >> "%OUT%" echo ===== 6. git show HEAD -p --stat (diffs empty) =====
  git show --name-only --pretty=format: HEAD > _names_show.txt 2>&1
  git show HEAD --stat --pretty=fuller >> "%OUT%" 2>&1
  >> "%OUT%" echo.
  for /f "usebackq delims=" %%F in ("_names_show.txt") do (
    if not "%%F"=="" git show HEAD -p --pretty=format: -- "%%F" >> "%OUT%" 2>&1
  )
)
del _names_head.txt _names_cached.txt _names_range.txt _names_show.txt 2>nul
echo WROTE %OUT%
for %%A in ("%OUT%") do echo bytes=%%~zA
