"""Live integration test for /report endpoint."""
import asyncio
import httpx
import sys

BASE = "http://localhost:8000"

PAYLOAD = {
    "claim_reference": "CP-TEST001",
    "vehicle_company": "Maruti",
    "vehicle_model": "Swift",
    "vehicle_year": "2019",
    "vehicle_variant": "VXI",
    "cause_of_damage": "Front collision with scraping on bumper",
    "component": "Front Bumper",
    "damage_type": "scratch",
    "severity": "moderate",
    "repair_approach": "OpenAI GPT-4o Vision",
    "explanation": "Visible linear scratches and scuff marks along the front bumper edge with paint transfer.",
    "cost_min": 3000,
    "cost_max": 6000,
    "settlement_offer": 3150,
    "settlement_min": 1650,
    "settlement_max": 5100,
    "final_settlement_amount": 5100,
    "gross_repair_min": 3000,
    "gross_repair_max": 6000,
    "depreciation_amount": 338,
    "depreciation_percentage": 8,
    "depreciation_reason": "Vehicle age 1-4 years",
    "deductible_amount": 1000,
    "net_settlement": 3162,
    "breakdown_items": [
        {"category": "Gross Repair Cost", "description": "Estimated repair", "amount": 4500, "type": "addition"},
        {"category": "Depreciation", "description": "8% vehicle age", "amount": -338, "type": "deduction"},
        {"category": "Policy Deductible", "description": "Standard", "amount": -1000, "type": "deduction"},
        {"category": "Net Cash Settlement", "description": "Final offer", "amount": 3162, "type": "addition"},
    ],
    "exclusions": [],
    "chat_history": [
        {"sender": "bot",  "text": "Based on assessment, we offer ₹3,150 for the front bumper scratch."},
        {"sender": "user", "text": "Can you offer more? The shop quoted ₹5,500."},
        {"sender": "bot",  "text": "I can go up to the maximum of ₹5,100. Does that work?"},
        {"sender": "user", "text": "I accept the offer."},
        {"sender": "bot",  "text": "Great! Settlement confirmed at ₹5,100. Please click Confirm Settlement."},
    ],
    "assessment_views": {
        "top":    {"detection_count": 1, "detections": [{"damage_type": "scratch", "confidence": 0.90, "bounding_box_xyxy": []}]},
        "bottom": {"detection_count": 1, "detections": [{"damage_type": "scratch", "confidence": 0.85, "bounding_box_xyxy": []}]},
        "left":   {"detection_count": 0, "detections": []},
        "right":  {"detection_count": 0, "detections": []},
    },
    "damage_items": [
        {
            "component": "Front Bumper",
            "damageType": "scratch",
            "severity": "moderate",
            "status": "confirmed",
            "recommendation": "repair",
            "costEstimateMin": 3000,
            "costEstimateMax": 6000,
        }
    ],
}


async def run():
    print("=" * 60)
    print("LIVE REPORT TEST — /report endpoint")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Sending to /report ...")
        resp = await client.post(f"{BASE}/report", json=PAYLOAD)
        print(f"Status: {resp.status_code}")

        if resp.status_code != 200:
            print("ERROR:", resp.text[:600])
            return False

        r = resp.json()
        print()
        print("claim_reference :", r.get("claim_reference"))
        print("generated_at    :", r.get("generated_at"))
        print("report length   :", len(r.get("report", "")), "chars")
        print("report_html len :", len(r.get("report_html", "")), "chars")
        print()
        print("--- Report preview (first 600 chars) ---")
        print(r.get("report", "")[:600])
        print()

        checks = [
            ("claim_reference present", bool(r.get("claim_reference")), r.get("claim_reference")),
            ("report not empty", len(r.get("report", "")) > 100, len(r.get("report", ""))),
            ("report_html not empty", len(r.get("report_html", "")) > 100, len(r.get("report_html", ""))),
            ("report contains claim ref", "CP-TEST001" in r.get("report", ""), "CP-TEST001"),
            ("report contains vehicle", "Maruti" in r.get("report", "") or "Swift" in r.get("report", ""), "Maruti/Swift"),
            ("report contains settlement amount", "5,100" in r.get("report", "") or "5100" in r.get("report", ""), "₹5100"),
        ]

        print("=== CHECKS ===")
        all_passed = True
        for name, ok, detail in checks:
            sym = "PASS" if ok else "FAIL"
            print(f"  [{sym}] {name}  (got: {detail})")
            if not ok:
                all_passed = False

        print()
        print("Overall:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
        return all_passed


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
