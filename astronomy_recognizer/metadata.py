from __future__ import annotations

from pathlib import Path

MAP_DATASET_ROOT = Path("Map_Dataset")

CATEGORY_DESCRIPTIONS = {
    "Galaxy": "A massive system of stars, gas, dust, and dark matter held together by gravity.",
    "Globular cluster": "A dense, spherical collection of very old stars orbiting the core of a galaxy.",
    "Open cluster": "A loose group of relatively young stars that formed from the same giant molecular cloud.",
    "Diffuse nebula": "A glowing cloud of gas and dust, often associated with active star formation.",
    "Planetary nebula": "An expanding shell of gas ejected by a dying star near the end of its life.",
    "Supernova remnant": "The expanding debris field left behind after a star explodes as a supernova.",
    "Other": "A Messier object that does not fit the major cluster, galaxy, or nebula categories.",
}


COMMON_NAMES = {
    "M1": "Crab Nebula",
    "M6": "Butterfly Cluster",
    "M7": "Ptolemy Cluster",
    "M8": "Lagoon Nebula",
    "M11": "Wild Duck Cluster",
    "M13": "Great Globular Cluster in Hercules",
    "M16": "Eagle Nebula",
    "M17": "Omega Nebula",
    "M20": "Trifid Nebula",
    "M24": "Sagittarius Star Cloud",
    "M27": "Dumbbell Nebula",
    "M31": "Andromeda Galaxy",
    "M33": "Triangulum Galaxy",
    "M40": "Winnecke 4",
    "M42": "Orion Nebula",
    "M43": "De Mairan's Nebula",
    "M44": "Beehive Cluster",
    "M45": "Pleiades",
    "M51": "Whirlpool Galaxy",
    "M57": "Ring Nebula",
    "M63": "Sunflower Galaxy",
    "M64": "Black Eye Galaxy",
    "M65": "Leo Triplet Galaxy",
    "M66": "Leo Triplet Galaxy",
    "M71": "Angelfish Cluster",
    "M76": "Little Dumbbell Nebula",
    "M81": "Bode's Galaxy",
    "M82": "Cigar Galaxy",
    "M97": "Owl Nebula",
    "M101": "Pinwheel Galaxy",
    "M102": "Spindle Galaxy",
    "M104": "Sombrero Galaxy",
    "M107": "Crucifix Cluster",
    "M109": "Vacuum Cleaner Galaxy",
    "M110": "Satellite of Andromeda",
}


def build_catalog_entry(image_path: Path) -> dict:
    object_id = image_path.stem.upper()
    category = image_path.parent.name.replace("_", " ")
    common_name = COMMON_NAMES.get(object_id)
    star_map = find_star_map(object_id)

    description = CATEGORY_DESCRIPTIONS.get(
        category,
        "A celestial object listed in the Messier catalog.",
    )
    if common_name:
        title = f"{object_id} ({common_name})"
    else:
        title = object_id

    return {
        "messier_id": object_id,
        "title": title,
        "common_name": common_name,
        "category": category,
        "description": description,
        "reference_image": str(image_path),
        "star_map_image": str(star_map) if star_map else None,
    }


def find_star_map(object_id: str) -> Path | None:
    if not MAP_DATASET_ROOT.exists():
        return None

    candidates = sorted(
        path
        for path in MAP_DATASET_ROOT.rglob("*")
        if path.is_file() and path.stem.upper() == object_id
    )
    return candidates[0] if candidates else None
