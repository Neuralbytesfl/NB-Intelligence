#!/usr/bin/env python3
import rumps
import psutil
import platform
import requests
import datetime
import re

# === Ollama API Config (your setup) ===
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "gpt-oss"


class MacAIHelper(rumps.App):
    def __init__(self):
        super(MacAIHelper, self).__init__("🤖 MacAI")
        self.menu = ["Run Health Check"]

    def collect_system_info(self):
        info = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory().percent,
            "memory_total": round(psutil.virtual_memory().total / (1024**3), 1),
            "memory_used": round(psutil.virtual_memory().used / (1024**3), 1),
            "disk_percent": psutil.disk_usage("/").percent,
            "disk_total": round(psutil.disk_usage("/").total / (1024**3), 1),
            "disk_used": round(psutil.disk_usage("/").used / (1024**3), 1),
            "uptime": self.get_uptime(),
        }
        return info

    def get_uptime(self):
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        delta = datetime.datetime.now() - boot_time
        return str(delta).split(".")[0]  # HH:MM:SS

    def sanitize_output(self, text: str) -> str:
        """Remove *, markdown symbols, and extra whitespace from AI reply."""
        clean = re.sub(r"[*`_>#-]", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def run_ai_check(self, sender):
        sys_info = self.collect_system_info()

        # Prompt for AI (to get human-readable interpretations)
        prompt = f"""
        You are a Mac health assistant. Summarize the system stats below into:
        1. A short interpretation of CPU, RAM, Disk, and Uptime.
        2. An overall Diagnosis.
        3. A Recommendation.
        
        Respond in plain sentences only, no markdown, no asterisks.

        System Data:
        {sys_info}
        """

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            reply = resp.json().get("response", "").strip()
        except Exception as e:
            reply = f"[Error: {e}]"

        reply = self.sanitize_output(reply)

        # --- Build formatted report ---
        report = f"""
==============================
🔎 Quick Look-through
==============================
CPU    : {sys_info['cpu_percent']}% usage
RAM    : {sys_info['memory']}% used ({sys_info['memory_used']} GB / {sys_info['memory_total']} GB)
Disk   : {sys_info['disk_percent']}% used ({sys_info['disk_used']} GB / {sys_info['disk_total']} GB)
Uptime : {sys_info['uptime']}

==============================
🩺 AI Diagnosis
==============================
{reply}

==============================
💡 Recommendation
==============================
Check CPU if consistently >80%
Check RAM if consistently >80%
Disk looks fine unless >90% used
==============================
"""
        # Show popup with Exit option
        result = rumps.alert("Mac AI Health Report", report, ok="OK", cancel="Exit App")
        if result == 0:  # Cancel pressed
            rumps.quit_application()

    @rumps.clicked("Run Health Check")
    def menu_health_check(self, sender):
        self.run_ai_check(sender)


if __name__ == "__main__":
    MacAIHelper().run()
