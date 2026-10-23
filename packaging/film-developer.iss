; Windows installer for film-developer (Inno Setup).
;
;     iscc packaging\film-developer.iss
;
; Build the payload first -- `pwsh packaging\make-win.ps1` lays down the
; embeddable Python, the wheels, the launcher and the icon under
; dist\film-developer\, and writes dist\version.txt. This wraps that folder
; into a Setup that installs to Program Files, drops a Start-menu shortcut, and
; registers an uninstaller. Same bytes the zip carries; this is the
; unpack-for-you face of it.
;
; There is no frozen exe, on purpose: PyInstaller's bootloader trips a Defender
; false positive (`Program:Win32/Vigram.A`) that truncated the old compiled
; installer to "setup files are corrupted". An embeddable Python is ordinary
; files, so Defender leaves it alone -- and this installer runs.
;
; **Not signed.** The Setup.exe still trips SmartScreen the first time -- a
; code-signing certificate is a yearly cost this project does not carry. The
; shortcut launches the windowed interpreter on the package; there is no console.

#define AppName "film-developer"
#define SrcDir  "dist\film-developer"
; The version from dist\version.txt, which make-win.ps1 wrote from the package
; -- one source, no second place to edit.
#define AppVer  Trim(FileRead(FileOpen("dist\version.txt")))

[Setup]
; A fixed AppId so upgrades replace in place instead of stacking installs.
AppId={{5B56886A-D03F-452D-8878-6699B867BFEA}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher=El Cacharrista
AppPublisherURL=https://elcacharrista.com
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; The GPL is shown at install time. Not a click-through EULA -- the GPL grants
; rather than restricts, and nobody has to accept it to *use* the program --
; but it is what this is under, so it is put in front of the person rather than
; buried in a subfolder they will never open.
LicenseFile=..\LICENSE
; The plain-text render, not the .md: Inno shows this verbatim and does not read
; Markdown, so the .md's headings, bold and table would leak their raw syntax
; here. make-win.ps1 writes the .txt beside the .md with md_to_text.py.
InfoAfterFile={#SrcDir}\licenses\THIRD-PARTY-LICENSES.txt
UninstallDisplayIcon={app}\film-developer.ico
; Per-user install by default: no admin prompt, lands in Local AppData.
PrivilegesRequiredOverridesAllowed=dialog
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=film-developer-{#AppVer}-x64-setup
Compression=lzma2
SolidCompression=no
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; The whole payload, recursively. recursesubdirs+createallsubdirs keeps the
; embeddable Python's Lib\site-packages tree intact -- imports fail without it.
; Excludes the bytecode caches. make-win.ps1 strips __pycache__ before it zips,
; so the zip is clean; but iscc runs later, and anything that executes the
; payload in between (a verification run, a curious double-click) writes fresh
; .pyc whose co_filename is the build checkout's absolute path -- the very path
; this build works to keep out. The .pyc are regenerated on first run, so
; excluding them costs nothing and makes the installer immune to that timing.
Source: "{#SrcDir}\*"; DestDir: "{app}"; Excludes: "*.pyc,*\__pycache__\*"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; The shortcut runs the windowed interpreter on the package -- no console, no
; launcher exe. The icon and working directory are set explicitly because
; pythonw.exe carries neither.
Name: "{group}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m film_developer"; WorkingDir: "{app}"; IconFilename: "{app}\film-developer.ico"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m film_developer"; WorkingDir: "{app}"; IconFilename: "{app}\film-developer.ico"; Tasks: desktopicon

[Run]
; Offer to launch on finish.
Filename: "{app}\python\pythonw.exe"; Parameters: "-m film_developer"; WorkingDir: "{app}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Python writes .pyc caches beside the code on first run; Inno only tracks what
; it installed, so without this the folder lingers full of bytecode after
; uninstall. Remove the whole tree.
Type: filesandordirs; Name: "{app}"
