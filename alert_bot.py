"""Scheduled alert bot (no Streamlit). Run by GitHub Actions, or locally:

    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python alert_bot.py
    DRY_RUN=1 python alert_bot.py        # print instead of sending

Config comes from config.json; env vars SYMBOLS / MIN_MATCH override it.
Last-seen matches are stored in state.json so you only hear about NEW matches.
"""
import json
import os
import sys
import time
from pathlib import Path

import ict_engine as eng

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "config.json"
STATE_FILE = ROOT / "state.json"


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main() -> int:
    cfg = load_json(CONFIG_FILE, {})
    symbols = [s.strip() for s in os.environ.get("SYMBOLS", "").split(",") if s.strip()]
    symbols = symbols or cfg.get("symbols", ["BTC-USD"])
    tfs = tuple(cfg.get("timeframes", ["Weekly", "Daily", "H4"]))
    min_match = int(os.environ.get("MIN_MATCH") or cfg.get("min_match", 1))
    cooldown = float(cfg.get("cooldown_hours", 4))
    strengths = cfg.get("strengths", {})

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    dry_run = os.environ.get("DRY_RUN") == "1"
    if not dry_run and not (token and chat_id):
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return 1

    state = load_json(STATE_FILE, {})
    failures = 0

    for symbol in symbols:
        frames = eng.fetch_tf_frames(symbol, tfs)
        matrix = eng.compute_matrix(frames, strengths)
        alerts, new_state = eng.decide_alerts(
            state.get(symbol, {}), matrix, tfs, min_match, cooldown
        )
        summary = {tf: (matrix.get(tf) or ("n/a",))[::2] for tf in tfs}
        print(f"{symbol}: {summary} -> {len(alerts)} alert(s)")

        for alert in alerts:
            text = eng.build_message(symbol, alert, matrix, tfs)
            if dry_run:
                print("--- DRY RUN ---\n" + text + "\n---------------")
                continue
            ok, err = eng.send_telegram(token, chat_id, text)
            if not ok:
                failures += 1
                print(f"Telegram failed for {symbol}: {err}", file=sys.stderr)
                # forget these keys so the next run retries
                for tf in alert["new"]:
                    key = f"{tf}:{alert['direction']}"
                    new_state["last_alert"].pop(key, None)
                    if key in new_state["matches"]:
                        new_state["matches"].remove(key)
        state[symbol] = new_state
        time.sleep(1)

    state["_last_run"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
