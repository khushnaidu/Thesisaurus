"""
Phase 1.3: Database Population
Takes the extracted CSV data and puts it into a SQLite database
Author: Student working on NLP project for robotics paper analysis
"""

import sqlite3
import csv
import sys
from pathlib import Path

# TODO: maybe switch to PostgreSQL later if we need more features?
# SQLite is good enough for now though

def create_database(db_path="papers.db"):
    """
    Create the database and all the tables we need
    This took me forever to get the schema right lol
    """
    print(f"Creating database at: {db_path}")
    
    # Delete old database if it exists (fresh start)
    if Path(db_path).exists():
        print("  → Found old database, deleting it...")
        Path(db_path).unlink()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main papers table - stores all the paper metadata
    cursor.execute("""
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT UNIQUE NOT NULL,
            title TEXT,
            authors TEXT,
            year INTEGER,
            venue TEXT,
            training_data_size TEXT,
            model_name TEXT,
            model_architecture TEXT,
            model_size TEXT,
            vision_encoder TEXT,
            base_model TEXT,
            optimizer TEXT,
            learning_rate TEXT,
            batch_size TEXT,
            epochs TEXT,
            augmentations TEXT,
            pretrained_weights TEXT,
            simulation_env TEXT,
            ml_framework TEXT,
            success_rate TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Datasets table - many-to-many relationship with papers
    cursor.execute("""
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE paper_datasets (
            paper_id TEXT,
            dataset_id INTEGER,
            dataset_type TEXT,  -- 'training' or 'evaluation'
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)
    
    # Robot platforms table
    cursor.execute("""
        CREATE TABLE robots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE paper_robots (
            paper_id TEXT,
            robot_id INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY (robot_id) REFERENCES robots(id)
        )
    """)
    
    # Hardware table (sensors, grippers, etc)
    cursor.execute("""
        CREATE TABLE hardware (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT  -- 'robot' for sensors/grippers, 'compute' for GPUs
        )
    """)
    
    cursor.execute("""
        CREATE TABLE paper_hardware (
            paper_id TEXT,
            hardware_id INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware(id)
        )
    """)
    
    # Baseline models table (for comparisons)
    cursor.execute("""
        CREATE TABLE baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE paper_baselines (
            paper_id TEXT,
            baseline_id INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY (baseline_id) REFERENCES baselines(id)
        )
    """)
    
    # Tasks table
    cursor.execute("""
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            task_description TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
        )
    """)
    
    # Create some indexes for faster queries
    # learned this is important from database class
    cursor.execute("CREATE INDEX idx_paper_id ON papers(paper_id)")
    cursor.execute("CREATE INDEX idx_dataset_name ON datasets(name)")
    cursor.execute("CREATE INDEX idx_robot_name ON robots(name)")
    
    conn.commit()
    print("  ✓ Database schema created!")
    return conn


def insert_paper_data(conn, row):
    """
    Insert a single paper's data into the database
    row is a dict from the CSV
    """
    cursor = conn.cursor()
    
    # Insert main paper info
    cursor.execute("""
        INSERT INTO papers (
            paper_id, title, authors, year, venue,
            training_data_size, model_name, model_architecture,
            model_size, vision_encoder, base_model,
            optimizer, learning_rate, batch_size, epochs,
            augmentations, pretrained_weights, simulation_env,
            ml_framework, success_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['paper_id'],
        row['title'],
        row['authors'],
        int(row['year']) if row['year'] else None,
        row['venue'] or None,
        row['training_data_size'] or None,
        row['model_name'] or None,
        row['model_architecture'] or None,
        row['model_size'] or None,
        row['vision_encoder'] or None,
        row['base_model'] or None,
        row['optimizer'] or None,
        row['learning_rate'] or None,
        row['batch_size'] or None,
        row['epochs'] or None,
        row['augmentations'] or None,
        row['pretrained_weights'] or None,
        row['simulation_env'] or None,
        row['ml_framework'] or None,
        row['success_rate'] or None
    ))
    
    paper_id = row['paper_id']
    
    # Handle datasets - training
    if row['training_datasets']:
        datasets = [d.strip() for d in row['training_datasets'].split(';') if d.strip()]
        for dataset in datasets:
            # Insert dataset if not exists
            cursor.execute("INSERT OR IGNORE INTO datasets (name) VALUES (?)", (dataset,))
            # Get dataset id
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset,))
            dataset_id = cursor.fetchone()[0]
            # Link paper to dataset
            cursor.execute(
                "INSERT INTO paper_datasets (paper_id, dataset_id, dataset_type) VALUES (?, ?, ?)",
                (paper_id, dataset_id, 'training')
            )
    
    # Handle datasets - evaluation
    if row['evaluation_datasets']:
        datasets = [d.strip() for d in row['evaluation_datasets'].split(';') if d.strip()]
        for dataset in datasets:
            cursor.execute("INSERT OR IGNORE INTO datasets (name) VALUES (?)", (dataset,))
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset,))
            dataset_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO paper_datasets (paper_id, dataset_id, dataset_type) VALUES (?, ?, ?)",
                (paper_id, dataset_id, 'evaluation')
            )
    
    # Handle robot platforms
    if row['robot_platforms']:
        robots = [r.strip() for r in row['robot_platforms'].split(';') if r.strip()]
        for robot in robots:
            cursor.execute("INSERT OR IGNORE INTO robots (name) VALUES (?)", (robot,))
            cursor.execute("SELECT id FROM robots WHERE name = ?", (robot,))
            robot_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO paper_robots (paper_id, robot_id) VALUES (?, ?)",
                (paper_id, robot_id)
            )
    
    # Handle robot hardware (sensors, grippers)
    if row['robot_hardware']:
        hardware_items = [h.strip() for h in row['robot_hardware'].split(';') if h.strip()]
        for hw in hardware_items:
            cursor.execute("INSERT OR IGNORE INTO hardware (name, type) VALUES (?, ?)", (hw, 'robot'))
            cursor.execute("SELECT id FROM hardware WHERE name = ?", (hw,))
            hw_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO paper_hardware (paper_id, hardware_id) VALUES (?, ?)",
                (paper_id, hw_id)
            )
    
    # Handle compute hardware (GPUs, TPUs)
    if row['compute_hardware']:
        hardware_items = [h.strip() for h in row['compute_hardware'].split(';') if h.strip()]
        for hw in hardware_items:
            cursor.execute("INSERT OR IGNORE INTO hardware (name, type) VALUES (?, ?)", (hw, 'compute'))
            cursor.execute("SELECT id FROM hardware WHERE name = ?", (hw,))
            hw_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO paper_hardware (paper_id, hardware_id) VALUES (?, ?)",
                (paper_id, hw_id)
            )
    
    # Handle baselines
    if row['baselines_compared']:
        baselines = [b.strip() for b in row['baselines_compared'].split(';') if b.strip()]
        for baseline in baselines:
            cursor.execute("INSERT OR IGNORE INTO baselines (name) VALUES (?)", (baseline,))
            cursor.execute("SELECT id FROM baselines WHERE name = ?", (baseline,))
            baseline_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO paper_baselines (paper_id, baseline_id) VALUES (?, ?)",
                (paper_id, baseline_id)
            )
    
    # Handle tasks
    if row['tasks_evaluated']:
        tasks = [t.strip() for t in row['tasks_evaluated'].split(';') if t.strip()]
        for task in tasks:
            cursor.execute(
                "INSERT INTO tasks (paper_id, task_description) VALUES (?, ?)",
                (paper_id, task)
            )
    
    conn.commit()


def populate_from_csv(csv_path, db_path="papers.db"):
    """
    Main function to read CSV and populate the database
    """
    print(f"\n{'='*60}")
    print("Phase 1.3: Populating Database from CSV")
    print(f"{'='*60}\n")
    
    # Create database
    conn = create_database(db_path)
    
    # Read and insert data
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    print(f"\nReading CSV: {csv_path}")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        papers = list(reader)
    
    print(f"Found {len(papers)} papers to insert\n")
    
    # Insert each paper
    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] Inserting: {paper['paper_id']}")
        try:
            insert_paper_data(conn, paper)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    # Print some stats
    print(f"\n{'='*60}")
    print("Database populated successfully! 🎉")
    print(f"{'='*60}\n")
    
    cursor = conn.cursor()
    
    # Count stuff
    cursor.execute("SELECT COUNT(*) FROM papers")
    paper_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM datasets")
    dataset_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM robots")
    robot_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM hardware WHERE type='compute'")
    compute_hw_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM baselines")
    baseline_count = cursor.fetchone()[0]
    
    print(f"📊 Database Statistics:")
    print(f"  • Papers: {paper_count}")
    print(f"  • Unique datasets: {dataset_count}")
    print(f"  • Robot platforms: {robot_count}")
    print(f"  • Compute hardware types: {compute_hw_count}")
    print(f"  • Baseline models: {baseline_count}")
    
    # Show some example queries
    print(f"\n💡 Example queries you can run:")
    print(f"  - Most common datasets:")
    cursor.execute("""
        SELECT d.name, COUNT(*) as count 
        FROM datasets d
        JOIN paper_datasets pd ON d.id = pd.dataset_id
        GROUP BY d.name
        ORDER BY count DESC
        LIMIT 5
    """)
    print("    ", cursor.fetchall())
    
    print(f"\n  - Papers with model sizes:")
    cursor.execute("SELECT paper_id, model_name, model_size FROM papers WHERE model_size IS NOT NULL")
    print("    ", cursor.fetchall())
    
    conn.close()
    print(f"\n✓ Database saved to: {db_path}")
    print("\nYou can now query it with: sqlite3 {db_path}")


if __name__ == "__main__":
    # Default paths
    csv_file = "extracted_info.csv"
    db_file = "papers.db"
    
    # Allow command line args
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        db_file = sys.argv[2]
    
    populate_from_csv(csv_file, db_file)

