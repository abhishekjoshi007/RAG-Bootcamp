import json
from pathlib import Path

from src.eval_units import collect_curriculum_units, extract_curriculum_units


def test_extract_curriculum_units_from_program_payload():
    payload = {
        "university": "Texas A&M University",
        "short_name": "TAMU",
        "college": "College of Engineering",
        "curriculum_year": 2024,
        "programs": [
            {
                "field": "Computer Science",
                "degree": "B.S. Computer Science",
                "department": "CSCE",
                "units": [
                    {
                        "id": "tamu_cs_ai",
                        "title": "Generative AI",
                        "description": "Transformers and RAG.",
                        "courses": ["CSCE 689"],
                        "current_topics": ["transformers", "RAG"],
                        "cs2023_area": "Intelligent Systems",
                    }
                ],
            }
        ],
    }

    units = extract_curriculum_units(payload, source_path=Path("data/universities/tamu_engineering.json"))

    assert len(units) == 1
    assert units[0]["id"] == "tamu_cs_ai"
    assert units[0]["metadata"]["field"] == "Computer Science"
    assert units[0]["metadata"]["source_file"] == "data/universities/tamu_engineering.json"


def test_collect_curriculum_units_skips_duplicate_ids(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    payload = {
        "university": "U",
        "units": [
            {
                "id": "unit_1",
                "title": "Unit 1",
                "description": "",
                "current_topics": [],
            }
        ],
    }
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))

    units, duplicates = collect_curriculum_units([second, first])

    assert [unit["id"] for unit in units] == ["unit_1"]
    assert duplicates == ["unit_1"]
