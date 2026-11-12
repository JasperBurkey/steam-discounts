# Registers the hourly collector tasks and keeps the PC awake on AC power.
# Run once from the project folder in an Administrator PowerShell:
#   .\setup_windows.ps1                      (collector tasks only)
#   .\setup_windows.ps1 -BackupDir E:\steam_backup   (also a nightly backup copy)
param([string]$BackupDir = "")
$ErrorActionPreference = "Stop"
$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "This must run in PowerShell opened as Administrator (right-click PowerShell > Run as administrator)." -ForegroundColor Red
  exit 1
}
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat = Join-Path $here "run_group.bat"

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Hours 20) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
# S4U = runs whether or not you are logged in, without storing your password.
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited

function HourlyTrigger([int]$minute) {
  $at = (Get-Date).Date.AddMinutes($minute)
  $t = New-ScheduledTaskTrigger -Daily -At $at
  $rep = New-ScheduledTaskTrigger -Once -At $at -RepetitionInterval (New-TimeSpan -Hours 1) `
         -RepetitionDuration (New-TimeSpan -Hours 23 -Minutes 59)
  $t.Repetition = $rep.Repetition
  return $t
}

foreach ($g in @(@("fast", 5), @("daily", 10), @("ccu", 15))) {
  $action = New-ScheduledTaskAction -Execute $bat -Argument $g[0] -WorkingDirectory $here
  Register-ScheduledTask -TaskName ("SteamCollector-" + $g[0]) -TaskPath "\SteamCollector\" -Action $action `
    -Trigger (HourlyTrigger $g[1]) -Settings $settings -Principal $principal -Force -ErrorAction Stop | Out-Null
  Write-Host ("Registered SteamCollector-{0}: every hour at :{1:D2}" -f $g[0], $g[1])
}

if ($BackupDir) {
  New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
  $dataDir = Join-Path $here "data"
  # /E copies new and changed files and never deletes anything at the destination.
  $roboArgs = "`"$dataDir`" `"$BackupDir`" /E /XO /R:2 /W:5 /NP /LOG+:`"$dataDir\logs\backup.log`""
  $action = New-ScheduledTaskAction -Execute "robocopy.exe" -Argument $roboArgs
  $trigger = New-ScheduledTaskTrigger -Daily -At "03:30"
  Register-ScheduledTask -TaskName "SteamCollector-backup" -TaskPath "\SteamCollector\" -Action $action `
    -Trigger $trigger -Settings $settings -Principal $principal -Force -ErrorAction Stop | Out-Null
  Write-Host "Registered SteamCollector-backup: daily 03:30 local, copying data to $BackupDir"
}

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
Write-Host "Sleep and hibernate disabled while plugged in."
Write-Host ""
Get-ScheduledTask -TaskPath "\SteamCollector\" | Format-Table TaskName, State -AutoSize
Write-Host "Check: Task Scheduler > Task Scheduler Library > SteamCollector"
