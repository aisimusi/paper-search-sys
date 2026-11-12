from __future__ import annotations

import re
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path


def unit_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def write_schedule(config: dict, folder: Path) -> tuple:
    if config["agent"]["backend"] == "codex-session":
        raise ValueError("codex-session 需要活跃的 Codex 会话；独立定时任务请选择 codex 或 harness backend")
    if os.name == "nt":
        return write_windows_schedule(config, folder)
    return write_systemd_schedule(config, folder)


def write_systemd_schedule(config: dict, folder: Path) -> tuple:
    root = Path(config["_root"])
    hour, minute = config["schedule"]["local_time"].split(":")
    name = re.sub(r"[^a-zA-Z0-9_-]", "-", config["project"]["name"]).strip("-") or "research"
    unit_name = "scholar-daily-" + name
    codex = shutil.which("codex")
    command = [sys.executable, "-m", "scholar_daily", "run", "--config", config["_config_path"]]
    path = ":".join(dict.fromkeys([str(Path(sys.executable).parent), str(Path(codex).parent) if codex else "/usr/local/bin", "/usr/local/bin", "/usr/bin", "/bin"]))
    service = "\n".join([
        "[Unit]", "Description=Personal daily research: " + name, "Wants=network-online.target", "After=network-online.target", "",
        "[Service]", "Type=oneshot", "WorkingDirectory=" + str(root).replace("%", "%%"),
        "Environment=" + unit_quote("PATH=" + path), "Environment=PYTHONUNBUFFERED=1", "UMask=0077",
        "ExecStart=" + " ".join(unit_quote(part) for part in command), "TimeoutStartSec=3h", "",
    ])
    timer = "\n".join([
        "[Unit]", "Description=Daily literature survey timer", "", "[Timer]",
        "OnCalendar=*-*-* %s:%s:00 Asia/Shanghai" % (hour, minute), "Persistent=true", "RandomizedDelaySec=30",
        "Unit=" + unit_name + ".service", "", "[Install]", "WantedBy=timers.target", "",
    ])
    folder.mkdir(parents=True, exist_ok=True)
    service_path, timer_path = folder / (unit_name + ".service"), folder / (unit_name + ".timer")
    service_path.write_text(service, encoding="utf-8")
    timer_path.write_text(timer, encoding="utf-8")
    return service_path, timer_path


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_windows_schedule(config: dict, folder: Path) -> tuple:
    """Write an importable Task Scheduler definition; never register a task."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    root = Path(config["_root"])
    name = re.sub(r"[^a-zA-Z0-9_-]", "-", config["project"]["name"]).strip("-") or "research"
    stem = "scholar-daily-" + name
    runner, task = folder / (stem + ".ps1"), folder / (stem + ".xml")
    prefix = Path(sys.executable).parent
    paths = [prefix, prefix / "Library" / "bin", prefix / "Scripts"]
    runner.write_text("\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$env:PYTHONUTF8 = '1'", "$env:PYTHONIOENCODING = 'utf-8'",
        "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)",
        "$env:PATH = " + powershell_quote(";".join(map(str, paths)) + ";") + " + $env:PATH",
        "Set-Location -LiteralPath " + powershell_quote(str(root)),
        "& " + powershell_quote(sys.executable) + " -m scholar_daily run --config " + powershell_quote(config["_config_path"])
        + " *>> " + powershell_quote(str(folder / (stem + ".log"))),
        "exit $LASTEXITCODE", "",
    ]), encoding="utf-8-sig")
    ns = "http://schemas.microsoft.com/windows/2004/02/mit/task"
    ET.register_namespace("", ns)
    def node(parent, tag, text=None, **attributes):
        element = ET.SubElement(parent, "{" + ns + "}" + tag, attributes)
        if text is not None:
            element.text = text
        return element
    document = ET.Element("{" + ns + "}Task", {"version": "1.2"})
    registration = node(document, "RegistrationInfo")
    node(registration, "Description", "Scholar Daily (Asia/Shanghai)")
    triggers = node(document, "Triggers")
    trigger = node(triggers, "CalendarTrigger")
    hour, minute = map(int, config["schedule"]["local_time"].split(":"))
    now = datetime.now(timezone(timedelta(hours=8)))
    start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if start <= now:
        start += timedelta(days=1)
    node(trigger, "StartBoundary", start.isoformat())
    node(trigger, "Enabled", "true")
    node(node(trigger, "ScheduleByDay"), "DaysInterval", "1")
    principal = node(node(document, "Principals"), "Principal", id="Author")
    node(principal, "LogonType", "InteractiveToken")
    node(principal, "RunLevel", "LeastPrivilege")
    settings = node(document, "Settings")
    for tag, value in [("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
                       ("StopIfGoingOnBatteries", "false"), ("StartWhenAvailable", "true"),
                       ("ExecutionTimeLimit", "PT3H"), ("Enabled", "true")]:
        node(settings, tag, value)
    action = node(node(document, "Actions", Context="Author"), "Exec")
    node(action, "Command", str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"))
    node(action, "Arguments", subprocess.list2cmdline(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(runner)]))
    node(action, "WorkingDirectory", str(root))
    ET.ElementTree(document).write(task, encoding="utf-16", xml_declaration=True)
    return runner, task
