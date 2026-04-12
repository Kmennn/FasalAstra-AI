"""
One-time Kaggle auth setup for FasalAstra.
Writes KAGGLE_API_TOKEN to PowerShell profile so it persists across sessions.
"""
import os
import pathlib

TOKEN = "KGAT_ba692935d95b459536f8d91d671720f0"

ps_profile = (
    pathlib.Path(os.environ["USERPROFILE"])
    / "Documents"
    / "WindowsPowerShell"
    / "Microsoft.PowerShell_profile.ps1"
)
ps_profile.parent.mkdir(parents=True, exist_ok=True)

line = f'$env:KAGGLE_API_TOKEN="{TOKEN}"  # FasalAstra Kaggle auth\n'

content = ps_profile.read_text(encoding="utf-8") if ps_profile.exists() else ""
if "KAGGLE_API_TOKEN" not in content:
    with open(ps_profile, "a", encoding="utf-8") as f:
        f.write("\n" + line)
    print(f"✅ Added KAGGLE_API_TOKEN to: {ps_profile}")
    print("   Restart PowerShell once, then 'kaggle datasets list' will work.")
else:
    print(f"✅ KAGGLE_API_TOKEN already in PS profile: {ps_profile}")

print(f"\n   Token: {TOKEN[:12]}...{TOKEN[-6:]}")
print("\n   For THIS session, run:")
print(f'   $env:KAGGLE_API_TOKEN="{TOKEN}"')
