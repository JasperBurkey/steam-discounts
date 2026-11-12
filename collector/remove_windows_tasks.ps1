# Removes the collector's scheduled tasks (data is not touched). Run as Administrator.
Get-ScheduledTask -TaskPath "\SteamCollector\" -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false
Write-Host "SteamCollector tasks removed."
