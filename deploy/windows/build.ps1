# Builds the Windows installer.  Every step that fails stops the build: a setup file is only produced from a tested, self-checked bundle.
param([switch]$SkipTests)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
function Step($text, [scriptblock]$cmd) { Write-Host "== $text"; & $cmd; if ($LASTEXITCODE -ne 0) { throw "$text failed" } }

if (-not $SkipTests) {
    # the release is built on tests that RAN: an absent model, corpus or driving fixture fails here instead of skipping
    $env:VGPT_REQUIRE_TESTDATA = "driving_fixture,corpus,sample_cars,models"
    Step "tests (start-up, routes, worker, records, confidence, release, corpus, pipe, mapper, voice, app, ingest, onboarding, storage)" { python -m pytest tests/test_bootstrap.py tests/test_route_policy.py tests/test_worker_queue.py tests/test_recordstore.py tests/test_confidence.py tests/test_release_manifest.py tests/test_release_model.py tests/test_corpus.py tests/test_no_cache.py tests/test_fleet_cache.py tests/test_version.py tests/test_pipe.py tests/test_mapper.py tests/test_voice.py tests/test_app_install.py tests/test_ingest.py tests/test_onboard.py tests/test_linked.py tests/test_archive.py tests/test_aaos.py tests/test_isolation.py tests/test_instant_load.py tests/test_lin.py tests/test_doip.py tests/test_flexray.py tests/test_protocol_cars.py -q -p no:cacheprovider }
    Remove-Item Env:VGPT_REQUIRE_TESTDATA
}
Step "bundle (PyInstaller)" { python -m PyInstaller --noconfirm --clean --distpath deploy/windows/build --workpath deploy/windows/work deploy/windows/Vantage.spec }

$Bundle = Join-Path $PSScriptRoot "build\Vantage"
# the provenance of this release, inside the bundle (the program's own doctor and /api/health read it): source from the project, models and vocabulary from the bundle
Step "release manifest (source, models, vocabulary, test data, runtime)" { python tools_release_manifest.py make --root (Join-Path $Bundle "_internal") --source-root $Repo --out (Join-Path $Bundle "_internal\RELEASE.json") }
$Cli = Join-Path $Bundle "VantageCli.exe"
Step "bundle self-check (doctor)" { & $Cli doctor }
Step "bundle lists the simulated cars" { $n = (& $Cli car --list | Measure-Object -Line).Lines; if ($n -lt 45) { throw "only $n car models" } }
Step "bundle speech model and microphone library" { foreach ($p in @("_internal\models\asr\vosk-model-small-en-us-0.15", "_internal\vosk", "_internal\_sounddevice_data", "_internal\licenses\opendbc\LICENSE", "_internal\llama_cpp\lib\llama.dll", "_internal\llama_cpp\lib\ggml.dll", "_internal\data\world4\aster_ev\aster_ev_4yr.db", "_internal\vgpt\vehicle\protocol_demo\body.ldf")) { if (-not (Test-Path (Join-Path $Bundle $p))) { throw "missing $p" } } }

$Iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $Iscc)) { $Iscc = Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe" }
if (-not (Test-Path $Iscc)) { throw "Inno Setup 6 is required (winget install JRSoftware.InnoSetup)" }
Step "installer (Inno Setup)" { & $Iscc (Join-Path $PSScriptRoot "Vantage.iss") }
$Out = Get-ChildItem (Join-Path $PSScriptRoot "dist") -Filter "Vantage-Setup-*.exe" | Sort-Object LastWriteTime | Select-Object -Last 1
Write-Host ("Installer: " + $Out.FullName + "  (" + $Out.Length + " bytes)")
$h = (certutil -hashfile $Out.FullName SHA256 | Select-Object -Index 1)
Write-Host ("SHA-256: " + $h.Trim())
# the manifest beside the installer, now with the installer's own hash (the installer cannot contain its own hash)
python tools_release_manifest.py make --root (Join-Path $Bundle "_internal") --source-root $Repo --installer $Out.FullName --out (Join-Path $PSScriptRoot ("dist\RELEASE-" + ($Out.BaseName -replace "Vantage-Setup-","") + ".json"))
