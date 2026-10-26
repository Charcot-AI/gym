"""Deterministic text presentations of a laboratory report."""

from __future__ import annotations

import csv
import io
from dataclasses import replace
from datetime import date

from charcot_gym.synth.source_formats.facts import LabReport

LEVEL_FORMATS = {
    "f0": ("table", "key_value"),
    "f1": ("narrative",),
    "f2": ("csv", "hl7", "paged_text", "reordered"),
    "f3": ("superseded", "converted_units", "ambiguous_date", "same_name", "impossible"),
    "f4": ("paged_inbox", "attachment", "split_attachments"),
}


def _header(report: LabReport) -> str:
    return (
        f"Synthetic Laboratory Report | {report.accession}\n"
        f"Patient: {report.name} | DOB: {report.dob} | MRN: {report.mrn} | Sex: {report.sex}\n"
        f"Collected: {report.collected} | Reported: {report.reported}\n"
    )


def _table(report: LabReport) -> str:
    rows = ["LOINC | Test | Result | UCUM | Reference | Flag | Comment"]
    rows.extend(
        f"{r.code} | {r.display} | {r.value} | {r.unit} | {r.low}-{r.high} | {r.flag} | {r.comment}"
        for r in report.results
    )
    return _header(report) + "\n".join(rows) + "\n"


def _key_value(report: LabReport) -> str:
    return (
        _header(report)
        + "\n".join(
            f"{r.display} ({r.code}): {r.value} {r.unit}; reference {r.low}-{r.high}; "
            f"flag {r.flag}; comment {r.comment}"
            for r in report.results
        )
        + "\n"
    )


def _narrative(report: LabReport) -> str:
    return (
        _header(report)
        + "\n".join(
            f"The {r.display} (LOINC {r.code}) was {r.value} {r.unit} "
            f"(reference {r.low}-{r.high}; flag {r.flag}). Lab comment: {r.comment}"
            for r in report.results
        )
        + "\n"
    )


def _csv(report: LabReport) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["LOINC", "Test", "Result", "UCUM", "Low", "High", "Flag", "Comment"])
    for r in report.results:
        writer.writerow([r.code, r.display, r.value, r.unit, r.low, r.high, r.flag, r.comment])
    return _header(report) + stream.getvalue()


def _hl7(report: LabReport) -> str:
    collected = report.collected.replace("-", "")
    reported = report.reported.replace("-", "")
    given, family = report.name.split(" ", 1)
    rows = [
        f"MSH|^~\\&|SYNTHLAB|LAB|CHART|GYM|{reported}||ORU^R01|{report.accession}|P|2.5.1",
        f"PID|1||{report.mrn}||{family}^{given}||"
        f"{report.dob.replace('-', '')}|{report.sex[0].upper()}",
        f"OBR|1||{report.accession}|LAB^Laboratory panel|||{collected}"
        f"|||||||||||||||{reported}|||F",
    ]
    for index, r in enumerate(report.results, 1):
        rows.append(
            f"OBX|{index}|NM|{r.code}^{r.display}^LN||{r.value}|{r.unit}^^UCUM|"
            f"{r.low}-{r.high}|{r.flag}|||F|||{collected}"
        )
        rows.append(f"NTE|{index}||{r.comment}")
    return "\n".join(rows) + "\n"


def _paged_text(report: LabReport) -> str:
    pages = [report.results[:3], report.results[3:]]
    return "\f".join(
        _table(replace(report, results=rows))
        + f"Page {index} of 2 | {report.accession} | End of page\n"
        for index, rows in enumerate(pages, 1)
    )


def _reordered(report: LabReport) -> str:
    return (
        _header(report)
        + "UCUM | Reference | LOINC | Flag | Result | Test | Comment\n"
        + "\n".join(
            f"{r.unit} | {r.low}-{r.high} | {r.code} | {r.flag} | "
            f"{r.value} | {r.display} | {r.comment}"
            for r in report.results
        )
        + "\n"
    )


def render_report(report: LabReport, format: str) -> str:
    """Render facts in a named source format without consuming global randomness."""
    renderers = {
        "table": _table,
        "key_value": _key_value,
        "narrative": _narrative,
        "csv": _csv,
        "hl7": _hl7,
        "paged_text": _paged_text,
        "reordered": _reordered,
    }
    if format in renderers:
        return renderers[format](report)
    if format not in {name for formats in LEVEL_FORMATS.values() for name in formats}:
        raise ValueError(f"unknown source format: {format}")
    if not report.results:
        return _table(report)
    first, *rest = report.results
    if format == "superseded":
        earlier = replace(report, results=(replace(first, value=first.value + 10), *rest))
        return (
            "Report status: preliminary\n"
            + _table(earlier)
            + ("Report status: amended final\n" + _table(report))
        )
    if format == "converted_units":
        factors = {"2345-7": (18.016, "mmol/L"), "2160-0": (1 / 88.42, "umol/L")}
        rows = tuple(
            replace(
                r,
                value=round(r.value / factors[r.code][0], 6),
                unit=factors[r.code][1],
                low=r.low / factors[r.code][0],
                high=r.high / factors[r.code][0],
            )
            if r.code in factors
            else r
            for r in report.results
        )
        return _table(replace(report, results=rows))
    if format == "ambiguous_date":
        return _table(
            replace(
                report,
                collected=date.fromisoformat(report.collected).strftime("%d/%m/%Y"),
                reported=date.fromisoformat(report.reported).strftime("%d/%m/%Y"),
            )
        )
    if format == "same_name":
        foreign = replace(
            report,
            dob="1974-09-21",
            mrn=report.mrn + "-OTHER",
            results=(replace(first, value=first.value + 10),),
        )
        return _table(foreign) + _table(report)
    if format == "impossible":
        return _table(
            replace(
                report,
                results=tuple(
                    replace(r, value=impossible_result(report)[1], flag="H", comment="")
                    if r.code == impossible_result(report)[0]
                    else r
                    for r in report.results
                ),
            )
        )
    if format in LEVEL_FORMATS["f4"]:
        return _table(report)
    raise ValueError(f"unknown source format: {format}")


def impossible_result(report: LabReport) -> tuple[str, float]:
    """Select a seeded analyte and a physiologically impossible high value."""
    limits = {"2823-3": 44, "2345-7": 20000, "2160-0": 500}
    row = next(r for r in report.results if r.code in limits)
    return row.code, limits[row.code]
