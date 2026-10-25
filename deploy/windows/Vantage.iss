#define AppName "Vantage"
#define AppVersion "1.3.4"
#define BuildDir "build\Vantage"

[Setup]
AppId={{C80E8F10-5448-4AE4-92EB-77ED1C416A13}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Vantage
DefaultDirName={localappdata}\Programs\Vantage
DefaultGroupName=Vantage
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=Vantage-Setup-{#AppVersion}-x64
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=..\..\ui\modern\vantage.ico
UninstallDisplayIcon={app}\Vantage.exe
LicenseFile=..\..\PRIVACY.md
CloseApplications=yes

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Vantage"; Filename: "{app}\Vantage.exe"; Parameters: "run --linked"; WorkingDir: "{app}"; Comment: "The assistant: reads the car connected to the local pipe"
Name: "{group}\Car Simulator"; Filename: "{app}\Vantage.exe"; Parameters: "car-ui"; WorkingDir: "{app}"; Comment: "A simulated vehicle that sends its own bus traffic to Vantage (can also load a whole recorded history at once, for testing)"
Name: "{group}\Vantage with the recorded sample history"; Filename: "{app}\Vantage.exe"; Parameters: "run"; WorkingDir: "{app}"
Name: "{group}\Vehicle Receiver (text window, advanced)"; Filename: "{app}\VantageCli.exe"; Parameters: "receive"; WorkingDir: "{app}"
Name: "{group}\Check this installation"; Filename: "{cmd}"; Parameters: "/k """"{app}\VantageCli.exe"" doctor"""; WorkingDir: "{app}"
Name: "{group}\Uninstall Vantage"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Vantage"; Filename: "{app}\Vantage.exe"; Parameters: "run --linked"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autodesktop}\Vantage Car Simulator"; Filename: "{app}\Vantage.exe"; Parameters: "car-ui"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create desktop shortcuts for Vantage and the Car Simulator"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\VantageCli.exe"; Parameters: "doctor"; Description: "Verify the offline installation"; Flags: postinstall waituntilterminated skipifsilent
Filename: "{app}\Vantage.exe"; Parameters: "run --linked"; Description: "Start Vantage"; Flags: postinstall nowait skipifsilent

[Code]
function WebView2Present(): Boolean;
var v: String;
begin
  Result := RegQueryStringValue(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v)
         or RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
  if Result then Result := (v <> '') and (v <> '0.0.0.0');
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not WebView2Present() then
    if MsgBox('Vantage draws its window with the Microsoft WebView2 runtime, which comes with Windows 11 and current Windows 10 and was not found here.' + #13#10 + #13#10 +
              'Install it from Windows Update (or Microsoft''s WebView2 installer) before starting Vantage. Continue with the installation?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
end;
