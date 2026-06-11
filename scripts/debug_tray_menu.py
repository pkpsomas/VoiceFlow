"""Drive the live VoiceFlow tray icon with UI Automation to debug the menu.

Finds the VoiceFlow tray icon (taskbar or overflow), right-clicks it,
dumps the context menu items UIA sees, screenshots the open menu, then
clicks the "Audio Source" item and reports whether config.json changed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import ImageGrab
from pywinauto import Desktop

from voiceflow.utils.settings import config_path

OUT_DIR = Path(__file__).resolve().parent
SHOT = OUT_DIR / "_tray_menu_debug.png"


def read_source() -> str:
    try:
        return json.loads(config_path().read_text(encoding="utf-8")).get("audio_input_source", "<missing>")
    except Exception as e:
        return f"<config read error: {e}>"


def find_tray_icon():
    desktop = Desktop(backend="uia")
    # 1) visible icons on the taskbar
    for top in ("Shell_TrayWnd",):
        try:
            bar = desktop.window(class_name=top)
            for btn in bar.descendants(control_type="Button"):
                name = btn.window_text() or ""
                if "voiceflow" in name.lower():
                    return btn, "taskbar"
        except Exception:
            pass
    # 2) hidden icons: open the overflow flyout via the chevron
    try:
        bar = desktop.window(class_name="Shell_TrayWnd")
        for btn in bar.descendants(control_type="Button"):
            if "hidden icons" in (btn.window_text() or "").lower():
                btn.click_input()
                time.sleep(1.0)
                break
        for w in desktop.windows():
            if "overflow" in (w.window_text() or "").lower() or "overflow" in w.element_info.class_name.lower():
                for btn in w.descendants(control_type="Button"):
                    name = btn.window_text() or ""
                    if "voiceflow" in name.lower():
                        return btn, "overflow"
    except Exception as e:
        print(f"overflow scan failed: {e}")
    return None, None


def main() -> int:
    print(f"config source BEFORE: {read_source()}")

    icon, where = find_tray_icon()
    if icon is None:
        print("FAIL: VoiceFlow tray icon not found")
        return 1
    print(f"icon found in {where}: {icon.window_text()!r}")

    icon.click_input(button="right")
    time.sleep(1.2)

    desktop = Desktop(backend="uia")
    menu = None
    for w in desktop.windows():
        cls = w.element_info.class_name or ""
        if cls == "#32768":
            menu = w
            break
    if menu is None:
        print("FAIL: no popup menu window (#32768) appeared after right-click")
        ImageGrab.grab().save(SHOT)
        print(f"full-screen shot saved: {SHOT}")
        return 1

    rect = menu.rectangle()
    print(f"menu window: {rect}")
    items = menu.descendants(control_type="MenuItem")
    print(f"menu items seen by UIA: {len(items)}")
    target = None
    for it in items:
        name = it.window_text() or ""
        state = it.element_info.element.CurrentIsEnabled
        print(f"  - {name!r} enabled={state}")
        if name.lower().startswith("audio source"):
            target = it

    ImageGrab.grab(bbox=(rect.left - 10, rect.top - 10, rect.right + 10, rect.bottom + 10)).save(SHOT)
    print(f"menu screenshot saved: {SHOT}")

    if target is None:
        print("FAIL: 'Audio Source' item not in menu")
        return 1

    target.click_input()
    time.sleep(1.5)
    print(f"config source AFTER: {read_source()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
