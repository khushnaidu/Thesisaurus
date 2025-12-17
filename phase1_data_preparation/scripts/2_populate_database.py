"""
simplified db population - reads from clean pymupdf text
"""

import re
import json
import sqlite3
from pathlib import Path


DATASET_PATTERNS = [
    (r'\bOpen[-\s]?X[-\s]?Embodiment\b', 'Open-X Embodiment'),
    (r'\bBridge[-\s]?Data(?:\s+V2)?\b', 'BridgeData V2'),
    (r'\bDROID\b', 'DROID'),
    (r'\bFranka[-\s]?Kitchen\b', 'Franka Kitchen'),
    (r'\bMetaWorld\b', 'MetaWorld'),
    (r'\bRLBench\b', 'RLBench'),
    (r'\bCALVIN\b', 'CALVIN'),
    (r'\bRoboMimic\b', 'RoboMimic'),
    (r'\bLanguage[-\s]?Table\b', 'Language-Table'),
    (r'\bManiSkill\b', 'ManiSkill'),
    (r'\bEgo4D\b', 'Ego4D'),
    (r'\bSimpler[-\s]?Env\b', 'SimplerEnv'),
    (r'\bRoboSuite\b', 'RoboSuite'),
    (r'\bLibero\b', 'Libero'),
    (r'\bRoboNet\b', 'RoboNet'),
]

VISION_PATTERNS = [
    (r'\bDINOv2\b', 'DINOv2'),
    (r'\bDINO\b', 'DINO'),
    (r'\bSigLIP\b', 'SigLIP'),
    (r'\bCLIP\b', 'CLIP'),
    (r'\bEfficientNet\b', 'EfficientNet'),
    (r'\bResNet[-\s]?\d*\b', 'ResNet'),
    (r'\bViT[-\s]?[BLS]?\b', 'ViT'),
]

ROBOT_PATTERNS = [
    (r'\bFranka[-\s]?(?:Emika[-\s]?)?Panda\b', 'Franka Panda'),
    (r'\bWidowX\b', 'WidowX'),
    (r'\bGoogle Robot\b', 'Google Robot'),
    (r'\bUnitree[-\s]?H1\b', 'Unitree H1'),
    (r'\bUR5e?\b', 'UR5'),
    (r'\bKinova\b', 'Kinova'),
    (r'\bxArm\b', 'xArm'),
    (r'\bAloha\b', 'Aloha'),
    (r'\bMobile[-\s]?Aloha\b', 'Mobile Aloha'),
    (r'\bSawyer\b', 'Sawyer'),
    (r'\bFetch\b', 'Fetch'),
]

HARDWARE_PATTERNS = [
    (r'\bA100\b', 'A100', 'gpu'),
    (r'\bH100\b', 'H100', 'gpu'),
    (r'\bV100\b', 'V100', 'gpu'),
    (r'\bRTX[-\s]?4090\b', 'RTX 4090', 'gpu'),
    (r'\bRTX[-\s]?3090\b', 'RTX 3090', 'gpu'),
    (r'\bTPU\b', 'TPU', 'gpu'),
    (r'\bRealSense\b', 'RealSense', 'sensor'),
    (r'\bZED\b', 'ZED', 'sensor'),
    (r'\bKinect\b', 'Kinect', 'sensor'),
]


def find_matches(text, patterns):
    found = set()
    for item in patterns:
        pattern, name = item[0], item[1]
        if re.search(pattern, text, re.IGNORECASE):
            found.add(name)
    return list(found)


def extract_year(text, meta):
    if meta.get('year'):
        return meta['year']
    m = re.search(r'arXiv:(\d{2})\d{2}\.\d+', text)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r'\b(202[0-5])\b', text[:3000])
    return int(m.group(1)) if m else None


def create_db(db_path):
    if Path(db_path).exists():
        Path(db_path).unlink()

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            year INTEGER,
            venue TEXT,
            vision_encoder TEXT,
            model_name TEXT
        )
    """)

    c.execute("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    c.execute("CREATE TABLE paper_datasets (paper_id TEXT, dataset_id INTEGER)")
    c.execute("CREATE TABLE robots (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    c.execute("CREATE TABLE paper_robots (paper_id TEXT, robot_id INTEGER)")
    c.execute("CREATE TABLE hardware (id INTEGER PRIMARY KEY, name TEXT UNIQUE, type TEXT)")
    c.execute("CREATE TABLE paper_hardware (paper_id TEXT, hardware_id INTEGER)")

    conn.commit()
    return conn


def populate(data_dir, db_path):
    data_path = Path(data_dir)

    with open(data_path / "metadata.json") as f:
        metadata_list = json.load(f)

    print(f"found {len(metadata_list)} papers")

    conn = create_db(db_path)
    c = conn.cursor()

    datasets_cache = {}
    robots_cache = {}
    hardware_cache = {}

    for meta in metadata_list:
        paper_id = meta['paper_id']
        print(f"  {paper_id}")

        text_file = data_path / "full_text" / f"{paper_id}.txt"
        if not text_file.exists():
            continue

        text = text_file.read_text()

        vision = find_matches(text, VISION_PATTERNS)
        datasets = find_matches(text, DATASET_PATTERNS)
        robots = find_matches(text, ROBOT_PATTERNS)
        hardware = [(p[1], p[2]) for p in HARDWARE_PATTERNS if re.search(p[0], text, re.IGNORECASE)]

        year = extract_year(text, meta)
        title = meta.get('title', '')[:200]

        model_name = None
        models = ['RT-1', 'RT-2', 'OpenVLA', 'Octo', 'EgoMimic', 'DexCap', 'OKAMI', 'Habitat', 'ReBot']
        for m in models:
            if m.lower().replace('-', '') in paper_id.lower():
                model_name = m
                break

        c.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?)",
                  (paper_id, title, year, meta.get('venue'), ', '.join(vision), model_name))

        for ds in datasets:
            if ds not in datasets_cache:
                c.execute("INSERT OR IGNORE INTO datasets (name) VALUES (?)", (ds,))
                c.execute("SELECT id FROM datasets WHERE name=?", (ds,))
                datasets_cache[ds] = c.fetchone()[0]
            c.execute("INSERT INTO paper_datasets VALUES (?,?)", (paper_id, datasets_cache[ds]))

        for r in robots:
            if r not in robots_cache:
                c.execute("INSERT OR IGNORE INTO robots (name) VALUES (?)", (r,))
                c.execute("SELECT id FROM robots WHERE name=?", (r,))
                robots_cache[r] = c.fetchone()[0]
            c.execute("INSERT INTO paper_robots VALUES (?,?)", (paper_id, robots_cache[r]))

        for hw_name, hw_type in hardware:
            if hw_name not in hardware_cache:
                c.execute("INSERT OR IGNORE INTO hardware (name,type) VALUES (?,?)", (hw_name, hw_type))
                c.execute("SELECT id FROM hardware WHERE name=?", (hw_name,))
                hardware_cache[hw_name] = c.fetchone()[0]
            c.execute("INSERT INTO paper_hardware VALUES (?,?)", (paper_id, hardware_cache[hw_name]))

    conn.commit()

    c.execute("SELECT COUNT(*) FROM papers")
    print(f"\ndone: {c.fetchone()[0]} papers")
    c.execute("SELECT COUNT(*) FROM datasets")
    print(f"  {c.fetchone()[0]} datasets")
    c.execute("SELECT COUNT(*) FROM robots")
    print(f"  {c.fetchone()[0]} robots")

    conn.close()


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "../outputs/papers.db"

    print("populating database")
    print("=" * 40)
    populate(data_dir, db_path)
