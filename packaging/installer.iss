; Inno Setup 6 script for PartyMix - per-user install, no admin rights needed.
;
; Build (CI does this):
;   iscc /DAppVersion=0.1.0 /DBuildDir=..\dist\PartyMix /DOutputDir=..\dist packaging\installer.iss
; Produces  <OutputDir>\PartyMix-Setup-<AppVersion>.exe
;
; AppVersion may be a pre-release string such as "0.1.0-dev.42"; the numeric
; Windows file version is taken from AppVersionNumeric (defaults to 0.0.0.0).

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef AppVersionNumeric
  #define AppVersionNumeric "0.0.0.0"
#endif
#ifndef BuildDir
  #define BuildDir "..\dist\PartyMix"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

#define AppName "PartyMix"
#define AppExe "PartyMix.exe"
#define AppPublisher "Casper ten Broeke"
#define AppURL "https://github.com/caspertenbroeke/partymix"

[Setup]
; Fixed AppId: keep it identical across versions so upgrades replace the old install.
AppId={{7B6E4C2A-3F1D-4E8B-9A57-2C64D1F0B8E3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
VersionInfoVersion={#AppVersionNumeric}
; Per-user install: {autopf} -> %LOCALAPPDATA%\Programs, shortcuts in the user's start menu/desktop.
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
InfoBeforeFile=installer_info_nl.txt
OutputDir={#OutputDir}
OutputBaseFilename=PartyMix-Setup-{#AppVersion}
#if FileExists(AddBackslash(SourcePath) + "..\partymix\gui\assets\icon.ico")
SetupIconFile=..\partymix\gui\assets\icon.ico
#endif
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
ShowLanguageDialog=auto

[Languages]
Name: "nl"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Remove the previous version's program files before installing (avoids stale DLLs after upgrades).
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{autoprograms}\{#AppName} verwijderen"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The analysis cache (separated stems, can be several GB). Settings and finished mixes are kept.
Type: filesandordirs; Name: "{localappdata}\PartyMix\cache"
