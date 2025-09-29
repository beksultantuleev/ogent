"""
Analytics and logging utilities
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from ..core.config import settings


class AnalyticsLogger:
    """Handles session and query analytics logging"""

    def __init__(self):
        self.analytics_log = settings.analytics_log
        self.query_log = settings.query_log

    def extract_phone_mentions(self, text: str) -> List[str]:
        """Extract phone brand mentions from text"""
        phone_keywords = [
            "iphone", "samsung", "galaxy", "pixel", "xiaomi", "huawei",
            "oppo", "vivo", "oneplus", "realme", "nokia", "motorola",
            "sony", "lg", "asus", "poco", "honor"
        ]

        mentions = []
        text_lower = text.lower()
        for keyword in phone_keywords:
            if keyword in text_lower:
                mentions.append(keyword)
        return mentions

    def log_query(self, tool_name: str, query: str, results_count: int):
        """Log individual tool query"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "query": query,
            "results_count": results_count
        }

        self._append_to_json(self.query_log, entry)

    def log_session(self, queries: List[str], phone_mentions: List[str]):
        """Log complete session analytics"""
        session_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_queries": len(queries),
            "phone_mentions": phone_mentions,
            "unique_phones_mentioned": list(set(phone_mentions)),
            "queries": queries[-10:] if len(queries) > 10 else queries,
            "session_summary": " ".join(queries[-3:]) if queries else ""
        }

        self._append_to_json(self.analytics_log, session_entry)

    def _append_to_json(self, file_path: Path, entry: Dict[str, Any]):
        """Append entry to JSON file"""
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
        else:
            data = []

        data.append(entry)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get analytics statistics"""
        try:
            with open(self.analytics_log, "r", encoding="utf-8") as f:
                data = json.load(f)

            total_sessions = len(data)
            total_queries = sum(session["total_queries"] for session in data)
            all_phones = []
            for session in data:
                all_phones.extend(session["phone_mentions"])

            return {
                "total_sessions": total_sessions,
                "total_queries": total_queries,
                "most_mentioned_phones": self._get_top_mentions(all_phones),
                "latest_sessions": data[-5:] if len(data) > 5 else data
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return {"total_sessions": 0, "total_queries": 0}

    def _get_top_mentions(self, mentions: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top phone mentions"""
        if not mentions:
            return []

        from collections import Counter
        counts = Counter(mentions)
        return [{"phone": phone, "count": count}
                for phone, count in counts.most_common(top_n)]