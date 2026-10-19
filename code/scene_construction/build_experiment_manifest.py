"""Create the preregistered 2025 AIS date manifest.

The fixed calendar rule avoids choosing days after observing model results:
first/third Wednesday and second/fourth Sunday of every month.
"""

from __future__ import annotations

import calendar
import csv
from datetime import date
from pathlib import Path


PORTS = ("new_york", "houston_galveston", "chesapeake")


def nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    days = [day for day in calendar.Calendar().itermonthdates(year, month) if day.month == month and day.weekday() == weekday]
    return days[occurrence - 1]


def split(month: int) -> str:
    if month <= 8:
        return "train"
    if month <= 10:
        return "validation"
    return "blind_test"


def main() -> None:
    output = Path("outputs/ais_2025_preregistered_manifest.csv")
    rows = []
    for month in range(1, 13):
        selected = (
            (nth_weekday(2025, month, calendar.WEDNESDAY, 1), "weekday_1"),
            (nth_weekday(2025, month, calendar.WEDNESDAY, 3), "weekday_3"),
            (nth_weekday(2025, month, calendar.SUNDAY, 2), "weekend_2"),
            (nth_weekday(2025, month, calendar.SUNDAY, 4), "weekend_4"),
        )
        for day, stratum in sorted(selected):
            for port in PORTS:
                rows.append(
                    {
                        "date": day.isoformat(),
                        "month": month,
                        "weekday": day.strftime("%A"),
                        "calendar_stratum": stratum,
                        "port": port,
                        "ny_split": split(month) if port == "new_york" else "external_only",
                        "allowed_for_tuning": "yes" if port == "new_york" and month <= 10 else "no",
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} port-days ({len(rows)//len(PORTS)} dates) to {output}")


if __name__ == "__main__":
    main()
