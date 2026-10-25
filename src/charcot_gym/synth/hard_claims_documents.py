"""Render clinic, payer, pharmacy, and MSP documents."""

# Document builders keep each record's required fields explicit.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

from datetime import date
from typing import Any


def _rules_text(plan: dict[str, Any], fee_rows: dict[str, list[dict[str, Any]]]) -> str:
    lines = [f"Healthplan rules for {plan['name']}."]
    required = plan.get("prior_auth_required", [])
    if required:
        lines.append(
            f"Prior authorization is required for {', '.join(required)}; approval must be active "
            "on the service date and cover the administered units."
        )
    if plan.get("cms_jw_jz"):
        lines.append("This plan follows CMS JW/JZ reporting for single-dose containers.")
    else:
        lines.append(
            "This commercial plan does not adopt CMS JW/JZ single-dose-container reporting."
        )
    if plan.get("ndc_reporting_required"):
        lines.append("Report the vial's National Drug Code on every drug line.")
    for code, rows in fee_rows.items():
        lines.append(
            f"{code} fee schedule: {rows[0]['start']}={rows[0]['price']}; "
            f"{rows[1]['start']} onward={rows[1]['price']}."
        )
    return "\n".join(lines)


def _visit_text(
    regimen: str,
    cancer_code: str,
    patient_name: str,
    mrn: str,
    service_date: date,
) -> str:
    laterality = "right" if cancer_code == "C50.911" else "left"
    descriptions = {
        "early_breast": (
            f"Adjuvant infusion following surgery for {laterality}-sided breast malignancy."
        ),
        "metastatic_breast_bone": (
            f"Infusion visit for {laterality}-sided breast malignancy with osseous disease."
        ),
        "metastatic_prostate": (
            "Infusion visit for metastatic prostate malignancy; tolerating the current cycle."
        ),
    }
    return (
        "Riverside Oncology Clinic\n"
        "4100 Fictional Medical Plaza, Fairhaven, OR 97000 | (555) 010-2200\n"
        "ONCOLOGY INFUSION VISIT\n"
        f"Patient: {patient_name}     MRN: {mrn}     Date: {service_date.isoformat()}\n"
        "S: Here for the scheduled infusion cycle; no new infusion-related symptoms reported.\n"
        f"O: {descriptions[regimen]}\n"
        f"A: Active treatment for {cancer_code}.\n"
        "P: Proceed with the documented regimen; pharmacy compounding worksheet records "
        "prepared products and administration details."
    )


def _mar_text(
    administrations: list[dict[str, Any]],
    bsa: float,
    height_cm: int,
    weight_kg: float,
    patient_name: str,
    mrn: str,
    service_date: date,
    split: str,
    index: int,
) -> str:
    lines = [
        "Riverside Oncology Clinic — Infusion Pharmacy",
        "PHARMACY COMPOUNDING WORKSHEET / MEDICATION ADMINISTRATION RECORD",
        f"Patient: {patient_name}     MRN: {mrn}     Date of service: {service_date.isoformat()}",
        f"Height: {height_cm} cm     Weight: {weight_kg:.1f} kg     Mosteller BSA: {bsa:.3f} m²",
        f"Cycle: {next((item['cycle'] for item in administrations if item.get('cycle')), 'continuing')}",
        "",
        f"{'Drug':<16} {'Ordered dose':<16} {'Prepared dose':<16} {'Discarded':>12}",
        "-" * 64,
    ]
    for row_number, item in enumerate(administrations, 1):
        dose = int(item["dose"])
        lot = f"FIC-{split[0].upper()}{index:03d}-{row_number:02d}"
        discarded = int(item.get("waste_mg", 0)) if item["container_type"] == "single_dose" else 0
        if item["code"] == "J9171":
            vials = item.get("vials_mg", _vials_for(int(item["dose"])))
            labels = item["container_labels"]
            vial_lines = [
                f"    {item['name']} vial: {labels[size]} x{vials.count(size)}"
                + (f"; NDC {item['ndc_by_size'][size]}" if item.get("ndc_by_size", {}).get(size) else "")
                for size in sorted(set(vials), reverse=True)
            ]
            if item["container_type"] == "multi_dose":
                vial_lines[-1] += f", {dose} mg withdrawn"
            vial_lines[-1] += f"; lot {lot}"
        else:
            sizes = item["vials_mg"]
            labels = item["container_labels"]
            vial_lines = [
                f"    {item['name']} vial: {labels[size]} x{sizes.count(size)}"
                for size in sorted(set(sizes), reverse=True)
            ]
            if item["container_type"] == "multi_dose":
                vial_lines[0] += f", {dose} mg withdrawn"
        vial_size = max(item.get("vials_mg", [item["container_sizes_mg"][0]]))
        ndc = item.get("ndc_by_size", {}).get(vial_size)
        if ndc:
            if item["code"] != "J9171":
                vial_lines[0] += f"; NDC {ndc}"
        if item.get("cycle"):
            vial_lines[-1] += f"; cycle {item['cycle']}"
        vial_lines[-1] += f"; lot {lot}"
        dose_text = f"{dose} {item['dose_unit']}"
        lines.append(f"{item['name']:<16} {dose_text:<16} {dose_text:<16} {f'{discarded} mg':>12}")
        lines.extend(vial_lines)
    pharmacist = ("JL", "CM", "RC", "TB")[index % 4]
    lines.extend(
        [
            "",
            f"Pharmacist initials: {pharmacist}     Compounded: {service_date.isoformat()} 13:42",
            f"Administration completed: {service_date.isoformat()} 15:18",
            "All lot identifiers are fictitious.",
        ]
    )
    return "\n".join(lines)


def _msp_questionnaire_text(
    patient_name: str, mrn: str, status: str, employer_name: str, employee_count: int
) -> str:
    employed = status == "employed"
    employed_box = "[X] Yes [ ] No" if employed else "[ ] Yes [X] No"
    retired_since = status.removeprefix("retired since ") if not employed else ""
    return "\n".join(
        [
            "Riverside Oncology Clinic",
            "MEDICARE SECONDARY PAYER QUESTIONNAIRE",
            f"Patient: {patient_name}     MRN: {mrn}",
            f"Is coverage based on current employment? {employed_box}",
            "Is this an employer group plan? [X] Yes [ ] No",
            f"Retired since: {retired_since or ''}________________",
            f"Employer name: {employer_name}____________________________",
            f"Employer employee count: {employee_count}________________",
            "Signature: ______________________________  Date: ______________",
            "Please complete for coordination-of-benefits review.",
        ]
    )


def _remittance_text(
    payer: str,
    claim: dict[str, Any],
    patient_name: str,
    mrn: str,
    claim_number: str,
    service_date: date,
) -> str:
    lines = [
        f"{payer} — CLAIM REMITTANCE ADVICE",
        "Claims Processing Center",
        "100 Example Parkway, Fairhaven, OR 97000",
        f"Claim number: {claim_number}",
        f"Patient: {patient_name}     MRN: {mrn}",
        f"Date of service: {service_date.isoformat()}",
        "",
        "BILLED LINES",
        f"{'Code':<10} {'Units':>8} {'Billed':>12} {'Allowed':>12} {'Paid':>12}",
        "-" * 58,
    ]
    for item in claim["item"]:
        code = item["productOrService"]["coding"][0]["code"]
        amount = float(item["net"]["value"])
        lines.append(
            f"{code:<10} {int(item['quantity']['value']):>8} "
            f"{'$' + format(amount, ',.2f'):>12} "
            f"{'$0.00':>12} {'$0.00':>12}"
        )
    lines.extend(
        [
            "",
            "DENIAL REASON",
            "Reason: billed units do not match the documented administered dose.",
        ]
    )
    return "\n".join(lines)


def _vials_for(dose: int) -> list[int]:
    candidates = []
    for count_160 in range(dose // 160 + 2):
        for count_80 in range(dose // 80 + 2):
            for count_20 in range(dose // 20 + 2):
                vials = [160] * count_160 + [80] * count_80 + [20] * count_20
                total = sum(vials)
                if total >= dose and vials:
                    candidates.append(
                        (total - dose, len(vials), tuple(sorted(vials, reverse=True)))
                    )
    return list(min(candidates)[2])


def _vials_for_sizes(dose: int, sizes: list[int]) -> list[int]:
    candidates = []
    for count in range(1, dose // min(sizes) + 2):
        for combination in _combinations(sizes, count):
            total = sum(combination)
            if total >= dose:
                candidates.append((total - dose, count, tuple(sorted(combination, reverse=True))))
    return list(min(candidates)[2])


def _combinations(sizes: list[int], count: int) -> list[tuple[int, ...]]:
    if count == 0:
        return [()]
    return [
        (size, *tail)
        for size in sizes
        for tail in _combinations(sizes, count - 1)
        if size >= (tail[0] if tail else size)
    ]
