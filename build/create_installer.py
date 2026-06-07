from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
APP_NAME = "AirMouse"
APP_VERSION = "0.1.0"

DIST_DIR = REPO_ROOT / "dist"
DIST_APP_DIR = DIST_DIR / APP_NAME
INSTALLER_OUT_DIR = DIST_DIR / "installer"

BUILD_DIR = REPO_ROOT / ".pyinstaller" / "build"
SPEC_DIR = REPO_ROOT / ".pyinstaller"
SPEC_PATH = SPEC_DIR / "AirMouse.spec"
INSTALLER_DIR = SPEC_DIR / "installer"

MODELS_DIR = REPO_ROOT / "models"
ICON_PATH = REPO_ROOT / "assets" / "icon.ico"
ENTRY_POINT = REPO_ROOT / "airmouse" / "__main__.py"

REQUIRED_MODEL = MODELS_DIR / "gesture_model.xml"


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print(f"\n> {' '.join(str(x) for x in cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd))


def _require_path(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}\n{hint}")


def _require_models() -> None:
    _require_path(
        MODELS_DIR,
        "Create a 'models' folder in the project root with OpenVINO weights.",
    )
    if not REQUIRED_MODEL.is_file():
        raise FileNotFoundError(
            f"Required model file missing: {REQUIRED_MODEL}\n"
            "Export OpenVINO IR (gesture_model.xml + .bin) into models/ before building."
        )


def _sync_models(dist_app: Path) -> Path:
    """Copy models next to the exe (paths.WEIGHTS_DIR when frozen)."""
    _require_models()
    dst = dist_app / "models"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(MODELS_DIR, dst)
    print(f"Models synced: {MODELS_DIR} -> {dst}")
    return dst


def resolve_iscc() -> Path | None:
    found = shutil.which("iscc.exe")
    if found:
        return Path(found)
    for env_key in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_key, "")
        if not base:
            continue
        candidate = Path(base) / "Inno Setup 6" / "ISCC.exe"
        if candidate.is_file():
            return candidate
    return None


def _normalize_setup_icon(src: Path, dst: Path) -> Path:
    """Re-save icon so Inno Setup accepts it (some .ico variants fail)."""
    try:
        from PIL import Image
    except ImportError:
        print("WARNING: Pillow not installed; using original icon for installer.")
        return src

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGBA")
        sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        im.save(dst, format="ICO", sizes=sizes)
    return dst


def _pyinstaller_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(SPEC_DIR),
        "--log-level",
        "WARN",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "sklearn",
        "--exclude-module",
        "scikit_learn",
        "--exclude-module",
        "ml",
        "--exclude-module",
        "onnx",
        "--exclude-module",
        "onnxscript",
    ]

    if args.clean:
        cmd.append("--clean")

    if SPEC_PATH.exists() and not args.regen_spec:
        cmd.append(str(SPEC_PATH))
        return cmd

    _require_path(ENTRY_POINT, "Package entry point airmouse/__main__.py is missing.")

    cmd += [
        "--name",
        APP_NAME,
        "--windowed",
        "--onedir",
        "--paths",
        str(REPO_ROOT),
        "--add-data",
        f"{MODELS_DIR}{os.pathsep}models",
        "--hidden-import",
        "pygrabber.dshow_graph",
        "--collect-submodules",
        "mediapipe",
        "--collect-all",
        "openvino",
        "--collect-all",
        "cv2",
        "--collect-all",
        "PyQt5",
        "--collect-submodules",
        "pyqtgraph",
        "--collect-submodules",
        "pynput",
        "--collect-submodules",
        "pyautogui",
    ]

    if ICON_PATH.is_file():
        cmd += ["--icon", str(ICON_PATH)]

    if args.lite:
        cmd += [
            "--exclude-module",
            "matplotlib",
            "--exclude-module",
            "reportlab",
            "--exclude-module",
            "openpyxl",
        ]

    cmd.append(str(ENTRY_POINT))
    return cmd


def build_portable(args: argparse.Namespace) -> Path:
    _require_models()
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    _run(_pyinstaller_cmd(args))

    exe = DIST_APP_DIR / f"{APP_NAME}.exe"
    if not exe.is_file():
        raise FileNotFoundError(f"Build failed: {exe} was not created.")

    _sync_models(DIST_APP_DIR)
    print(f"\nPortable build OK: {DIST_APP_DIR}")
    return DIST_APP_DIR


def _write_inno_script(
    *,
    dist_app: Path,
    setup_icon: Path | None,
    app_version: str,
) -> Path:
    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    iss_path = INSTALLER_DIR / f"{APP_NAME}-installer.iss"

    # Forward slashes work in Inno Setup paths on Windows.
    dist_src = dist_app.as_posix()
    icon_line = ""
    if setup_icon and setup_icon.is_file():
        icon_line = f'SetupIconFile="{setup_icon.as_posix()}"'

    iss = f"""#define MyAppName "{APP_NAME}"
#define MyAppVersion "{app_version}"
#define MyAppPublisher "{APP_NAME} Team"
#define MyAppExeName "{APP_NAME}.exe"

[Setup]
AppId={{{{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}}}}
AppName={{#MyAppName}}
AppVersion={{#MyAppVersion}}
AppPublisher={{#MyAppPublisher}}
DefaultDirName={{autopf}}\\{{#MyAppName}}
DefaultGroupName={{#MyAppName}}
DisableProgramGroupPage=yes
OutputDir="{INSTALLER_OUT_DIR.as_posix()}"
OutputBaseFilename="{APP_NAME}-setup-{app_version}"
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
{icon_line}
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{dist_src}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{autoprograms}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"
Name: "{{autodesktop}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{{#MyAppExeName}}"; Description: "{{cm:LaunchProgram,{{#StringChange(MyAppName, '&', '&&')}}}}"; Flags: nowait postinstall skipifsilent
"""
    iss_path.write_text(iss, encoding="utf-8")
    return iss_path


def build_installer(args: argparse.Namespace) -> Path:
    dist_app = DIST_APP_DIR
    if not (dist_app / f"{APP_NAME}.exe").is_file():
        print("Portable build not found; building first...")
        build_portable(args)

    _sync_models(dist_app)

    setup_icon: Path | None = None
    if ICON_PATH.is_file():
        normalized = INSTALLER_DIR / "setup.ico"
        try:
            setup_icon = _normalize_setup_icon(ICON_PATH, normalized)
        except Exception as exc:
            print(f"WARNING: Could not normalize icon: {exc}")
            setup_icon = ICON_PATH

    iss_path = _write_inno_script(
        dist_app=dist_app,
        setup_icon=setup_icon,
        app_version=args.version,
    )

    iscc = resolve_iscc()
    if iscc is None:
        raise RuntimeError(
            "Inno Setup 6 (ISCC.exe) not found.\n"
            "Install from https://jrsoftware.org/isinfo.php and rerun:\n"
            f"  python {Path(__file__).name} installer"
        )

    INSTALLER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _run([str(iscc), str(iss_path)])

    setup_exe = INSTALLER_OUT_DIR / f"{APP_NAME}-setup-{args.version}.exe"
    if setup_exe.is_file():
        print(f"\nInstaller OK: {setup_exe}")
    else:
        print(f"\nInstaller build finished; check: {INSTALLER_OUT_DIR}")
    return setup_exe


def release(args: argparse.Namespace) -> None:
    build_portable(args)
    build_installer(args)


def clean(_: argparse.Namespace) -> None:
    for path in (BUILD_DIR, DIST_DIR, SPEC_DIR):
        if path.exists():
            print(f"Removing: {path}")
            shutil.rmtree(path, ignore_errors=True)


def run_built(_: argparse.Namespace) -> None:
    exe = DIST_APP_DIR / f"{APP_NAME}.exe"
    if not exe.is_file():
        raise SystemExit(
            f"Built exe not found: {exe}\nRun: python {Path(__file__).name} build"
        )
    _run([str(exe)])


def main(argv: list[str] | None = None) -> int:
    try:
        from airmouse import __version__ as pkg_version

        default_version = pkg_version
    except Exception:
        default_version = APP_VERSION

    parser = argparse.ArgumentParser(
        prog="@build_installer",
        description="Build AirMouse portable app and Windows 10/11 installer (PyInstaller + Inno Setup).",
    )
    parser.add_argument(
        "--version",
        default=default_version,
        help="Version string for the setup filename (default: airmouse.__version__)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--clean", action="store_true", help="Clean PyInstaller cache before building")
    common.add_argument("--regen-spec", action="store_true", help="Regenerate .spec instead of reusing it")
    common.add_argument(
        "--lite",
        action="store_true",
        help="Exclude matplotlib/reportlab/openpyxl (faster build, fewer report features)",
    )

    p_build = sub.add_parser("build", parents=[common], help="Build portable app into dist/AirMouse/")
    p_build.set_defaults(func=build_portable)

    p_installer = sub.add_parser(
        "installer",
        parents=[common],
        help="Create Windows setup exe (requires dist or runs build first)",
    )
    p_installer.set_defaults(func=build_installer)

    p_release = sub.add_parser(
        "release",
        parents=[common],
        help="Build portable app + Windows installer (recommended)",
    )
    p_release.set_defaults(func=release)

    p_clean = sub.add_parser("clean", help="Remove dist/ and .pyinstaller/")
    p_clean.set_defaults(func=clean)

    p_run = sub.add_parser("run", help="Run dist/AirMouse/AirMouse.exe")
    p_run.set_defaults(func=run_built)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
